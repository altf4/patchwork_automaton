//! Owned, contiguous arrays. Borrowed input descriptors are valid only during collate.
use crate::Result;

pub enum Data {
    Float(Vec<f32>),
    Int(Vec<i32>),
    Bool(Vec<u8>),
}
pub struct Array {
    pub key: usize,
    pub shape: Vec<usize>,
    pub data: Data,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct View {
    pub data: *const u8,
    pub len: usize,
    pub rank: usize,
    pub dims: [usize; 4],
    pub dtype: u32,
    pub key: u32,
}
pub struct Arrays {
    pub arrays: Vec<Array>,
    pub views: Vec<View>,
}
impl Array {
    pub fn floats(key: usize, shape: &[usize], data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            key,
            shape: shape.into(),
            data: Data::Float(data),
        }
    }
    pub fn ints(key: usize, shape: &[usize], data: Vec<i32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            key,
            shape: shape.into(),
            data: Data::Int(data),
        }
    }
    pub fn bools(key: usize, shape: &[usize], data: Vec<u8>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            key,
            shape: shape.into(),
            data: Data::Bool(data),
        }
    }
    fn view(&self) -> View {
        let mut dims = [0; 4];
        dims[..self.shape.len()].copy_from_slice(&self.shape);
        let (data, len, dtype) = match &self.data {
            Data::Float(v) => (v.as_ptr().cast(), v.len(), 0),
            Data::Int(v) => (v.as_ptr().cast(), v.len(), 1),
            Data::Bool(v) => (v.as_ptr(), v.len(), 2),
        };
        View {
            data,
            len,
            dtype,
            dims,
            rank: self.shape.len(),
            key: self.key as u32,
        }
    }
}
impl Arrays {
    pub fn new(arrays: Vec<Array>) -> Self {
        let views = arrays.iter().map(Array::view).collect();
        Self { arrays, views }
    }
}

/// Input descriptors name 14 contiguous arrays per example in schema order.
/// The caller holds the NumPy allocations alive throughout this synchronous call.
pub unsafe fn collate(views: &[View], batch: usize) -> Result<Arrays> {
    if batch == 0 || views.len() != batch * 14 {
        return Err("Empty or malformed batch".into());
    }
    let ranks = [2, 1, 1, 3, 2, 3, 1, 2, 2, 1, 1, 2, 1, 1];
    let types = [0, 1, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 1, 2];
    let mut shapes: Vec<Vec<usize>> = ranks.iter().map(|&r| vec![0; r]).collect();
    for row in views.chunks_exact(14) {
        for (k, v) in row.iter().enumerate() {
            if v.rank != ranks[k]
                || v.key as usize != k
                || v.dtype != types[k]
                || v.data.is_null()
                || v.len == 0
                || v.len != v.dims[..v.rank].iter().product::<usize>()
            {
                return Err("Malformed tensor descriptor".into());
            }
            for (d, s) in shapes[k].iter_mut().enumerate() {
                *s = (*s).max(v.dims[d]);
            }
        }
        let n = row[0].dims[0];
        let a = row[7].dims[0];
        let s = row[3].dims[1];
        let h = row[11].dims[0];
        let expected = [
            vec![n, row[0].dims[1]],
            vec![n],
            vec![n],
            vec![n, s, row[3].dims[2]],
            vec![n, s],
            vec![n, n, 4],
            vec![row[6].dims[0]],
            vec![a, row[7].dims[1]],
            vec![a, n],
            vec![a],
            vec![row[10].dims[0]],
            vec![h, row[11].dims[1]],
            vec![h],
            vec![h],
        ];
        if row.iter().zip(expected).any(|(v, s)| v.dims[..v.rank] != s) {
            return Err("Inconsistent entity/candidate/history axes".into());
        }
        for k in [0, 3, 6, 7, 10, 11] {
            if row[k].dims[row[k].rank - 1] != views[k].dims[views[k].rank - 1] {
                return Err("Cannot mix feature schemas in a batch".into());
            }
        }
        if !unsafe { std::slice::from_raw_parts(row[9].data, row[9].len) }
            .iter()
            .any(|&b| b != 0)
        {
            return Err("Decision has no allowed candidate".into());
        }
    }
    let mut out = Vec::new();
    for k in 0..14 {
        let shape = &shapes[k];
        let size: usize = shape.iter().product();
        let mut full_shape = vec![batch];
        full_shape.extend(shape);
        let data = match types[k] {
            0 => Data::Float(vec![0.; batch * size]),
            1 => Data::Int(vec![0; batch * size]),
            _ => Data::Bool(vec![0; batch * size]),
        };
        let mut array = Array {
            key: k,
            shape: full_shape,
            data,
        };
        let target: *mut u8 = match &mut array.data {
            Data::Float(v) => v.as_mut_ptr().cast(),
            Data::Int(v) => v.as_mut_ptr().cast(),
            Data::Bool(v) => v.as_mut_ptr(),
        };
        let element = if types[k] == 2 { 1 } else { 4 };
        for b in 0..batch {
            let v = &views[b * 14 + k];
            let last = v.dims[v.rank - 1];
            let rows = v.len / last;
            for r in 0..rows {
                let offset = match v.rank {
                    1 => 0,
                    2 => r * shape[1],
                    3 => (r / v.dims[1]) * shape[1] * shape[2] + (r % v.dims[1]) * shape[2],
                    _ => unreachable!(),
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        v.data.add(r * last * element),
                        target.add((b * size + offset) * element),
                        last * element,
                    );
                }
            }
        }
        out.push(array);
    }
    Ok(Arrays::new(out))
}
