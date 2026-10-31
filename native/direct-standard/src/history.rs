//! Actor-private recurrent history derived from typed redacted observations.
use crate::{Encoder,Result,arrays::{Array,Arrays},entity};
use rustmage::{Observation,Zone};
use std::collections::{HashMap,HashSet};
const SEM:usize=252;
const WIDTH:usize=292;
const ZONES:[&str;12]=["unknown","player","hand","battlefield","graveyard","exile","stack","command","sideboard","revealed","looked_at","library"];
fn zone(z:Zone)->&'static str {match z {Zone::Library=>"library",Zone::Hand=>"hand",Zone::Battlefield=>"battlefield",Zone::Stack=>"stack",Zone::Graveyard=>"graveyard",Zone::Exile=>"exile",Zone::Command=>"command",Zone::Outside=>"outside"}}
fn hot(row:&mut Vec<f32>,v:&str,names:&[&str]) {row.extend(names.iter().map(|n|(*n==v) as u8 as f32));}
#[derive(Clone)]
struct Object {id:u32,identity:String,zone:&'static str,side:usize,traits:Vec<f32>}
struct State {revision:u64,objects:Vec<Object>,players:[(i128,usize,usize);2],turn:u32}
#[derive(Default)]
pub struct History { previous:Option<State>,pending:Vec<Vec<f32>> }
impl History {
    fn event(&mut self,kind:usize,side:usize,old:&str,new:&str,value:Option<f64>,delta:Option<f64>,turn:u32,traits:Option<&[f32]>)->Result<()> {
        if self.pending.len()>=4096 {return Err("Neural history overflow".into());}
        let mut row=Vec::with_capacity(WIDTH);
        entity::hot(&mut row,Some(kind),7);entity::hot(&mut row,Some(side),3);
        hot(&mut row,old,&ZONES);hot(&mut row,new,&ZONES);
        entity::number(&mut row,value);entity::number(&mut row,delta);entity::number(&mut row,Some(turn as f64));
        if let Some(t)=traits {row.extend(t);}else{row.extend([0.;SEM]);}
        if row.len()!=WIDTH {return Err("history width mismatch".into());}
        self.pending.push(row);Ok(())
    }
    pub fn observe(&mut self,obs:&Observation,encoder:&Encoder)->Result<()> {
        if self.previous.as_ref().is_some_and(|p|p.revision>=obs.revision) {return Ok(());}
        let objects:Vec<_>=obs.objects.iter().map(|o|{
            let name=o.card.identity();
            let identity=encoder.catalogue.identities.get(name).cloned().unwrap_or_else(||name.to_string());
            let traits=encoder.catalogue.traits.get(name).cloned().unwrap_or_else(||encoder.catalogue.unknown_traits.clone());
            Object {id:o.id,identity,zone:zone(o.zone),side:(o.controller!=obs.viewer) as usize,traits}
        }).collect();
        let players=obs.players.each_ref().map(|p|(p.life,p.hand_count,p.library_count));
        let previous=self.previous.take();
        let old:HashMap<_,_>=previous.as_ref().map(|p|p.objects.iter().map(|o|(o.id,o)).collect()).unwrap_or_default();
        for o in &objects {
            match old.get(&o.id) {
                Some(before) if before.identity==o.identity => if before.zone!=o.zone {
                    self.event(2,o.side,before.zone,o.zone,None,None,obs.turn,Some(&o.traits))?;
                },
                _=>self.event(0,o.side,"unknown",o.zone,None,None,obs.turn,Some(&o.traits))?,
            }
        }
        let current:HashSet<_>=objects.iter().map(|o|o.id).collect();
        if let Some(prev)=&previous {for o in &prev.objects {if !current.contains(&o.id) {
            self.event(1,o.side,o.zone,"unknown",None,None,obs.turn,Some(&o.traits))?;
        }}}
        for (i,p) in players.iter().enumerate() {
            let vals=[p.0,p.1 as i128,p.2 as i128];
            let before=previous.as_ref().map(|p|{let x=p.players[i];[x.0,x.1 as i128,x.2 as i128]});
            for (stat,value) in vals.into_iter().enumerate() {
                let old=before.map(|a|a[stat]);
                if old!=Some(value) { self.event(3+stat,(obs.viewer.index()!=i) as usize,"unknown","unknown",Some(value as f64),old.map(|x|value.checked_sub(x).map_or(value as f64 - x as f64, |v| v as f64)),obs.turn,None)?; }
            }
        }
        if previous.as_ref().is_none_or(|p|p.turn!=obs.turn) {self.event(6,2,"unknown","unknown",Some(obs.turn as f64),None,obs.turn,None)?;}
        self.previous=Some(State{revision:obs.revision,objects,players,turn:obs.turn});Ok(())
    }
    pub fn take(&mut self)->Arrays {
        let rows=std::mem::take(&mut self.pending); let size=rows.len().max(1);
        let mut values=vec![0.;size*WIDTH];let mut mask=vec![0;size];
        for (i,row) in rows.into_iter().enumerate(){values[i*WIDTH..(i+1)*WIDTH].copy_from_slice(&row);mask[i]=1;}
        Arrays::new(vec![Array::floats(11,&[size,WIDTH],values),Array::ints(12,&[size],vec![0;size]),Array::bools(13,&[size],mask)])
    }
}
