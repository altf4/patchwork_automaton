import unittest
from patchwork_automaton.backends.xmage_choices import object_views
from patchwork_automaton.backends.xmage_combat import combat_objects


class CombatTests(unittest.TestCase):
    def test_attackers_blockers_and_public_relationships_are_exposed(self):
        view=dict(myPlayerId='me',players=[dict(playerId='me',battlefield={
            'attacker':dict(id='attacker',name='Monastery Swiftspear')}),dict(playerId='other',battlefield={
            'blocker':dict(id='blocker',name='Goblin Guide')})])
        objects=object_views(view)
        rows=combat_objects(objects,[dict(attackers={'attacker':{}},blockers={'blocker':{}},defenderId='other',isBlocked=True)])
        by_id={h.value:o for h,o in rows.items()}
        self.assertTrue(by_id['attacker'].properties['attacking'])
        self.assertFalse(by_id['attacker'].properties['blocking'])
        self.assertIn('other',by_id['attacker'].properties['targets'])
        self.assertTrue(by_id['attacker'].properties['combatBlocked'])
        self.assertTrue(by_id['blocker'].properties['blocking'])
        self.assertIn('attacker',by_id['blocker'].properties['targets'])
        self.assertFalse(any('attacking' in o.properties for o in objects.values()))
        empty=combat_objects(objects,[])
        self.assertFalse(any(o.properties['attacking'] or o.properties['blocking'] for o in empty.values()))

    def test_missing_or_unrecognized_combat_does_not_invent_known_flags(self):
        objects=object_views(dict(myPlayerId='me',players=[dict(playerId='me',battlefield={
            'card':dict(id='card',name='Monastery Swiftspear')})]))
        for data in (None,{},[dict(attackers=None,blockers={})]):self.assertEqual(combat_objects(objects,data),objects)
