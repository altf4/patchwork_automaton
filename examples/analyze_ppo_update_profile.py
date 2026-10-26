"""Analyze saved timing probes, NVIDIA samples and TensorFlow XPlane traces."""
from collections import defaultdict
import csv
from datetime import datetime
import gzip
import json
import os
import re
from pathlib import Path
from zoneinfo import ZoneInfo


def merge_intervals(values):
    out=[]
    for a,b in sorted(values):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out


def duration(values):return sum(b-a for a,b in merge_intervals(values))


def intersect(a,b):
    a=merge_intervals(a);b=merge_intervals(b);out=[];i=j=0
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if hi>lo:out.append([lo,hi])
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out


def invert(values,start,end):
    out=[];cursor=start
    for a,b in intersect(values,[(start,end)]):
        if a>cursor:out.append([cursor,a])
        cursor=max(cursor,b)
    if cursor<end:out.append([cursor,end])
    return out


def bin_occupancy(values,start,end,count):
    width=(end-start)/count;totals=[0.0]*count
    for lo,hi in intersect(values,[(start,end)]):
        index=max(0,min(count-1,int((lo-start)/width)))
        while index<count and lo<hi:
            stop=min(hi,start+(index+1)*width)
            if stop>lo:totals[index]+=stop-lo
            lo=stop;index+=1
    return [min(100.,value/width*100) for value in totals]


def nvidia_samples(path,started_wall_ns):
    rows=[]
    for raw in csv.reader(Path(path).read_text().splitlines()):
        if len(raw)!=8:continue
        try:
            t=datetime.strptime(raw[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=ZoneInfo('America/Phoenix')).timestamp()
            def number(v):
                try:return float(v)
                except ValueError:return None
            values=[number(v.strip()) for v in raw[1:]]
            rows.append({'elapsed_seconds':t-started_wall_ns/1e9,'gpu_busy_percent':values[0],'memory_busy_percent':values[1],
                         'vram_used_mib':values[2],'vram_total_mib':values[3],'power_watts':values[4],'sm_clock_mhz':values[5],'memory_clock_mhz':values[6]})
        except ValueError:continue
    return rows


def gpu_events(paths):
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
    rows=[];planes=[];errors=[]
    for path in paths:
        space=xplane_pb2.XSpace();space.ParseFromString(Path(path).read_bytes());errors.extend(space.errors);errors.extend(space.warnings)
        epoch_ns=None
        for environment in space.planes:
            if environment.name=='Task Environment':
                for stat in environment.stats:
                    if environment.stat_metadata[stat.metadata_id].name=='profile_start_time':epoch_ns=stat.uint64_value
        for plane in space.planes:
            if 'GPU' not in plane.name:continue
            lines=[]
            selected={meta.name:key for key,meta in plane.stat_metadata.items() if meta.name in ('tf_op','memcpy_details','kernel_details','core_count')}
            references={}
            def resolve(stat):
                if stat.WhichOneof('value')=='ref_value':
                    if stat.ref_value not in references:references[stat.ref_value]=plane.stat_metadata[stat.ref_value].name
                    return references[stat.ref_value]
                return stat.str_value
            sm_count=next((stat.int64_value for stat in plane.stats if stat.metadata_id==selected.get('core_count')),None)
            for line in plane.lines:
                # Derived TensorFlow/NVTX/group rows can span gaps between CUDA
                # activities. Select raw stream rows only, never their envelopes.
                raw=line.id>=0 and ('stream' in line.name.lower())
                lines.append({'id':line.id,'name':line.name,'events':len(line.events),'raw_stream_selected':raw})
                if not raw:continue
                for event in line.events:
                    name=plane.event_metadata[event.metadata_id].name
                    clock_base=epoch_ns/1e9 if epoch_ns and line.timestamp_ns<epoch_ns/2 else 0.0
                    start=clock_base+line.timestamp_ns/1e9+event.offset_ps/1e12
                    seconds=event.duration_ps/1e12
                    if seconds<=0:continue
                    lower=name.lower()
                    kind='copy' if 'memcpy' in lower else 'memset' if 'memset' in lower else 'kernel'
                    op='';copy_bytes=None;grid=None
                    for stat in event.stats:
                        if stat.metadata_id==selected.get('tf_op'):op=resolve(stat)
                        elif kind=='copy' and stat.metadata_id==selected.get('memcpy_details'):
                            found=re.search(r'size:(\d+)',resolve(stat));copy_bytes=int(found[1]) if found else None
                        elif kind=='kernel' and stat.metadata_id==selected.get('kernel_details'):
                            found=re.search(r'grid:(\d+),(\d+),(\d+)',resolve(stat));grid=int(found[1])*int(found[2])*int(found[3]) if found else None
                    direction=('device_to_host' if 'd2h' in lower or 'dtoh' in lower else 'host_to_device' if 'h2d' in lower or 'htod' in lower else 'device_to_device' if 'd2d' in lower or 'dtod' in lower else 'other') if kind=='copy' else None
                    rows.append({'start':start,'end':start+seconds,'seconds':seconds,'name':name,'kind':kind,'plane':plane.name,'line':line.name,
                                 'tf_op':op,'copy_bytes':copy_bytes,'copy_direction':direction,'grid_blocks':grid,'sm_count':sm_count})
            planes.append({'name':plane.name,'lines':lines,'task_environment_profile_start_ns':epoch_ns,'clock_alignment':'Relative XPlane timestamps offset by recorded profile_start_time when present; absolute timestamps retained.'})
    return rows,planes,errors


def analyze(root,output):
    import numpy as np
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    control=json.loads((root/'control/report.json').read_text());profile=json.loads((root/'profile/report.json').read_text())
    assert control['status']==profile['status']=='complete'
    initial_equal=all(control[k]==profile[k] for k in ('config','input_hashes','initial_model_sha256','initial_rng','source_hashes'))
    if not initial_equal:raise ValueError('Starting workload differs; do not attribute timings to instrumentation')
    keys=['samples','accepted_steps','rejected_epochs','critic_steps','optimizer_steps','critic_optimizer_steps']
    counters={k:[control['result'][k],profile['result'][k]] for k in keys}
    with np.load(root/'control/result-state.npz') as a,np.load(root/'profile/result-state.npz') as b:
        bitwise=all(np.array_equal(a[k],b[k]) for k in a.files)
        close=all(np.allclose(a[k],b[k],rtol=1e-7,atol=1e-7) for k in a.files)
        maximum=max(float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a.files)
    parity={'initial_state_and_inputs_equal':initial_equal,'final_rng_equal':control['final_rng']==profile['final_rng'],
            'bitwise_equal':bitwise,'allclose_1e_7':close,'max_absolute_difference':maximum,'counters':counters,
            'workload_counts_equal':all(x==y for x,y in counters.values()),'control_seconds':control['update_seconds'],'profile_seconds':profile['update_seconds'],
            'note':'Strict parity gate failed if counters or arrays differ. Elapsed-time ratio then is not a measurement of instrumentation overhead.'}
    samples={name:nvidia_samples(root/name/'nvidia-smi.csv',report['update_started_wall_ns']) for name,report in [('control',control),('profile',profile)]}
    statistics={}
    for name,data in samples.items():
        values=[x['gpu_busy_percent'] for x in data if x['gpu_busy_percent'] is not None]
        statistics[name]={'samples':len(values),'mean_gpu_busy_percent':float(np.mean(values)),'median_gpu_busy_percent':float(np.median(values)),'p95_gpu_busy_percent':float(np.quantile(values,.95)),
                          'peak_vram_used_mib':max(x['vram_used_mib'] for x in data if x['vram_used_mib'] is not None)}
    with gzip.open(root/'profile/host-spans.json.gz','rt') as f:host=json.load(f)
    trace=profile['profiling']['trace'];files=list((root/'profile/tf-trace').rglob('*.xplane.pb'))
    gpu,planes,errors=gpu_events(files)
    trace_summary={'available':bool(gpu),'files':[str(p.relative_to(root)) for p in files],'planes':planes,'errors':errors}
    host_windows={};window=None
    if gpu and trace.get('start_wall_ns'):
        start=trace['start_wall_ns']/1e9;end=min(start+trace['requested_seconds'],trace['stop_wall_ns']/1e9);window=(start,end)
        all_gpu=intersect([(g['start'],g['end']) for g in gpu],[window]);idle=invert(all_gpu,start,end)
        if not all_gpu:raise ValueError('GPU trace clock does not overlap the measured window; do not report zero activity')
        for name,tid,thread,a,b,cpu in host['events']:
            offset=host['t0_wall_ns']/1e9-host['t0_perf_ns']/1e9
            host_windows.setdefault(name,[]).append((a/1e9+offset,b/1e9+offset))
        overlaps={name:duration(intersect(intervals,idle)) for name,intervals in host_windows.items()}
        kernel_stats=defaultdict(lambda:[0.,0])
        for g in gpu:
            if g['kind']!='kernel':continue
            contribution=duration(intersect([(g['start'],g['end'])],[window]))
            if contribution:kernel_stats[g['name']][0]+=contribution;kernel_stats[g['name']][1]+=1
        top=sorted(({'name':name,'summed_kernel_seconds':s,'calls':count,'mean_kernel_us':s/count*1e6} for name,(s,count) in kernel_stats.items()),key=lambda g:-g['summed_kernel_seconds'])
        copies=defaultdict(lambda:{'calls':0,'bytes':0,'calls_at_most_16_bytes':0});copy_sources=defaultdict(lambda:[0,0]);gpu_ops=defaultdict(lambda:[0.,0]);kernel_durations=[];small_grid=known_grid=0
        for g in gpu:
            seconds=max(0.,min(g['end'],end)-max(g['start'],start))
            if not seconds:continue
            if g['kind']=='copy':
                group=copies[g['copy_direction']];group['calls']+=1
                if g['copy_bytes'] is not None:
                    group['bytes']+=g['copy_bytes'];group['calls_at_most_16_bytes']+=int(g['copy_bytes']<=16)
                if g['copy_direction']=='device_to_host':
                    source=copy_sources[g['tf_op'] or '(unattributed)'];source[0]+=1;source[1]+=g['copy_bytes'] or 0
            if g['kind']=='kernel':
                kernel_durations.append(g['seconds']*1e6);gpu_ops[g['tf_op'] or '(unattributed)'][0]+=seconds;gpu_ops[g['tf_op'] or '(unattributed)'][1]+=1
                if g['grid_blocks'] is not None and g['sm_count']:
                    known_grid+=1;small_grid+=int(g['grid_blocks']<g['sm_count'])
        trace_summary.update(copies_by_direction=dict(copies),top_device_to_host_sources=sorted(({'tf_op':op,'calls':v[0],'bytes':v[1]} for op,v in copy_sources.items()),key=lambda r:-r['calls'])[:15],
            top_gpu_ops=sorted(({'tf_op':op,'summed_kernel_seconds':v[0],'calls':v[1]} for op,v in gpu_ops.items()),key=lambda r:-r['summed_kernel_seconds'])[:20],
            kernel_duration_us_quantiles=dict(zip(['p50','p90','p99'],map(float,np.quantile(kernel_durations,[.5,.9,.99])))),known_kernel_grids=known_grid,kernels_with_fewer_blocks_than_sms=small_grid)
        trace_summary.update(window_start_wall_seconds=start,window_seconds=end-start,gpu_active_union_seconds=duration(all_gpu),gpu_active_fraction=duration(all_gpu)/(end-start),gpu_idle_seconds=duration(idle),
            kernel_active_union_seconds=duration(intersect([(g['start'],g['end']) for g in gpu if g['kind']=='kernel'],[window])),copy_active_union_seconds=duration(intersect([(g['start'],g['end']) for g in gpu if g['kind']=='copy'],[window])),
            kernel_calls=sum(g['calls'] for g in top),top_kernels=top[:20],gpu_idle_overlap_host_span_seconds=overlaps,
            note='Raw stream activity union, not theoretical compute utilization. Host-span overlaps are not additive or proof of causation.')
    result={'status':'complete' if parity['allclose_1e_7'] and parity['workload_counts_equal'] else 'complete_with_parity_limit','parity':parity,'nvidia_samples':statistics,'host_spans':profile['profiling']['spans'],'gpu_trace':trace_summary,
            'settings_unchanged':control['config']==profile['config'],'source_unchanged':control['source_unchanged'] and profile['source_unchanged'],
            'limitations':['One frozen round replay; no playing-strength comparison.','Instrumentation overhead is reported against the uninstrumented replay.','NVIDIA GPU utilization measures sampled activity, not FLOPS/SM occupancy.','Timings overlap across threads and nesting; do not sum them into a wall-time pie.']}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(output/'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(12,7.6),gridspec_kw={'height_ratios':[1.05,1]});fig.subplots_adjust(left=.24,right=.97,top=.84,bottom=.12,hspace=.52)
    fig.suptitle('GPU update pipeline · unchanged model and RL settings',x=.04,ha='left',fontsize=18,fontweight='bold',y=.97)
    fig.text(.04,.9,f'Control {control["update_seconds"]:.1f}s ({control["result"]["rejected_epochs"]} retries) · instrumented {profile["update_seconds"]:.1f}s ({profile["result"]["rejected_epochs"]} retries) · instrumented timing is not a speedup comparison',fontsize=11,color='#53667C')
    a=axes[0]
    for name,color,label in [('control','#2167A8','Uninstrumented control'),('profile','#A8B5C4','Instrumented replay')]:
        data=samples[name];a.plot([r['elapsed_seconds'] for r in data],[r['gpu_busy_percent'] for r in data],color=color,lw=1,alpha=.85,label=label)
    a.set_ylim(0,105);a.set_ylabel('Sampled GPU activity (%)');a.set_xlabel('Seconds into the update');a.grid(alpha=.15);a.legend(loc='upper right',frameon=False)
    chosen=[('initial_rollout_scan','Initial rollout scan'),('record_gather','Record loading / gather'),('collate','Batch array assembly'),('packing_lock_wait','Packing-lock waits'),('actor_batch_wait','Actor input-queue waits'),('critic_batch_wait','Critic input-queue waits'),('step_call','Actor TensorFlow calls'),('critic_step_call','Critic TensorFlow calls'),('guard_scan','Complete guard scans'),('restore_call','Actor/Adam rollback calls')]
    spans=profile['profiling']['spans'];names=[label for key,label in chosen];values=[spans.get(key,{}).get('wall_union_seconds',0) for key,label in chosen]
    a=axes[1];a.barh(names,values,color=['#147D70' if 'wait' not in key else '#B96C25' for key,label in chosen]);a.invert_yaxis();a.set_xlabel('Union of observed wall-time intervals (seconds; overlapping categories)');a.grid(axis='x',alpha=.15)
    for i,value in enumerate(values):a.text(value+.5,i,f'{value:.1f}s',va='center',fontsize=9)
    a.set_xlim(0,max(values)*1.17)
    fig.text(.04,.035,'GPU samples cover the whole update. CPU spans come from the instrumented pass and overlap; these bars are not additive.',fontsize=10,color='#53667C')
    fig.savefig(output/'pipeline-profile.png',dpi=160);fig.savefig(output/'pipeline-profile.svg');plt.close(fig)
    if window:
        start,end=window;fig,axes=plt.subplots(2,1,figsize=(12,6),gridspec_kw={'height_ratios':[1,1.4]});fig.subplots_adjust(left=.23,right=.97,top=.8,bottom=.15,hspace=.3)
        fig.suptitle(f'GPU activity and CPU waits · {end-start:.0f}-second trace',x=.04,ha='left',fontsize=18,fontweight='bold',y=.96)
        fig.text(.04,.86,f'Raw GPU stream activity: {100*trace_summary["gpu_active_fraction"]:.1f}% of the window · {trace_summary["kernel_calls"]:,} kernel calls',fontsize=12,color='#53667C')
        bins=np.linspace(start,end,301);activity=bin_occupancy(all_gpu,start,end,300)
        axes[0].stairs(activity,bins-start,color='#2167A8',fill=True,alpha=.8);axes[0].set_ylabel('GPU activity (%)');axes[0].set_ylim(0,105)
        categories=[('actor_batch_wait','Actor input wait'),('critic_batch_wait','Critic input wait'),('record_gather','Record gather'),('collate','Array assembly'),('packing_lock_wait','Packing-lock wait'),('guard_scan','Guard scan')]
        for i,(key,label) in enumerate(categories):
            intervals=intersect(host_windows.get(key,[]),[window]);axes[1].broken_barh([(a-start,b-a) for a,b in intervals],(i-.32,.64),facecolors='#B96C25' if 'wait' in key else '#147D70')
        axes[1].set_yticks(range(len(categories)),[x[1] for x in categories]);axes[1].invert_yaxis();axes[1].set_xlabel('Seconds from trace start')
        for a in axes:a.set_xlim(0,end-start);a.grid(axis='x',alpha=.15)
        fig.text(.04,.04,'Activity is busy time, not compute capacity. CPU rows overlap; coincident waits identify candidates rather than proving causation.',fontsize=10,color='#53667C')
        fig.savefig(output/'gpu-waits.png',dpi=160);fig.savefig(output/'gpu-waits.svg');plt.close(fig)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=analyze(a.input,a.output)
    print(json.dumps({'gpu_samples':r['nvidia_samples'],'gpu_trace_available':r['gpu_trace']['available'],'parity':r['parity']}))
