"""Expose structured public combat groups on existing visible objects."""
from collections.abc import Mapping
from dataclasses import replace


def combat_objects(objects,combat):
    if not isinstance(combat,(list,tuple)):return objects
    if any(not isinstance(g,Mapping) or not isinstance(g.get('attackers'),Mapping)
           or not isinstance(g.get('blockers'),Mapping) for g in combat):return objects
    attackers={};blockers={}
    for group in combat:
        for key in group['attackers']:
            attackers[key]=(group.get('defenderId'),group.get('isBlocked'))
        for key in group['blockers']:blockers.setdefault(key,[]).extend(group['attackers'])
    result={}
    for handle,obj in objects.items():
        if obj.zone!='battlefield':result[handle]=obj;continue
        props=dict(obj.properties,attacking=handle.value in attackers,blocking=handle.value in blockers)
        original=props.get('targets') or ()
        targets=[]
        if handle.value in attackers:
            defender,blocked=attackers[handle.value]
            if isinstance(defender,str):targets.append(defender)
            if type(blocked) is bool:props['combatBlocked']=blocked
        if handle.value in blockers:targets.extend(blockers[handle.value])
        if targets:
            props['combatTargets']=tuple(dict.fromkeys(targets))
            if isinstance(original,(list,tuple)) and all(isinstance(v,str) for v in original):
                props['targets']=tuple(dict.fromkeys((*original,*targets)))
        result[handle]=replace(obj,properties=props)
    return result
