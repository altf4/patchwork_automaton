"""Burn protocol identities survive UUID/order changes and reach numeric logits."""
from dataclasses import replace
import unittest
from uuid import UUID

from test_neural import AVAILABLE, observation
from patchwork_automaton.backends.xmage_choices import choices, encode_response
from patchwork_automaton.burn_semantics import CARD_REGISTRY
from patchwork_automaton.core import Handle, ObjectView


SOURCE = str(UUID(int=100))
OTHER = str(UUID(int=101))


def fixture(name, labels, *, second=None, hidden=False, source=SOURCE):
    view = {'myPlayerId': 'me', 'players': [{'playerId': 'me', 'battlefield': {
        SOURCE: {'id': SOURCE, 'name': name, 'faceDown': hidden},
        **({OTHER: {'id': OTHER, 'name': second}} if second else {})}}]}
    data = {'choices': {str(UUID(int=i+1)): label for i, label in enumerate(labels)}}
    options, bindings = choices('GAME_CHOOSE_ABILITY', data, view, source=source)
    obs = observation()
    objects = {Handle('xmage', 'object', key): ObjectView(Handle('xmage', 'object', key),
        raw['name'], 'battlefield', obs.actor, raw) for key, raw in view['players'][0]['battlefield'].items()}
    return replace(obs, objects=objects, decision=replace(obs.decision,
        kind='GAME_CHOOSE_ABILITY', options=tuple(options))), bindings


class BindingTests(unittest.TestCase):
    def test_mana_clicks_exclude_tapped_burn_lands_and_nonmana_objects(self):
        from uuid import UUID
        names=['Mountain','Sacred Foundry','Sunbaked Canyon','Barbarian Ring','Goblin Guide','Arid Mesa','Unreviewed mana source']
        for advertised in (False,True):
            cards={str(UUID(int=i+300)):dict(id=str(UUID(int=i+300)),name=name,tapped=True)
                   for i,name in enumerate(names)}
            raw=dict(myPlayerId='me',players=[dict(playerId='me',battlefield=cards)])
            if advertised:raw['canPlayObjects']={'objects':{k:{'basicManaAbilities':[{'id':'mana'}]} for k in cards}}
            options,_=choices('GAME_PLAY_MANA',{},raw)
            self.assertEqual({o.label for o in options},{'Unreviewed mana source','Done / Cancel'})
            options,_=choices('GAME_PLAY_MANA',{'targets':list(cards)},raw)
            self.assertEqual({o.label for o in options},{'Unreviewed mana source','Done / Cancel'})
            for c in cards.values():c['tapped']=False
            options,_=choices('GAME_PLAY_MANA',{},raw)
            self.assertEqual({o.label for o in options},set(names[:4])|{'Unreviewed mana source','Done / Cancel'})

    def test_damage_land_is_still_available_at_one_life(self):
        raw=dict(myPlayerId='me',players=[dict(playerId='me',life=1,battlefield={
            SOURCE:dict(id=SOURCE,name='Barbarian Ring',tapped=False)})])
        options,_=choices('GAME_PLAY_MANA',{},raw)
        self.assertIn('Barbarian Ring',{o.label for o in options})

    def test_reviewed_mana_costs_modes_and_fetch(self):
        cases = [('Sacred Foundry', ['7. {T}: Add {W}.', '2. {T}: Add {R}.'], [1, 0]),
            ('Sunbaked Canyon', ['{T}, Pay 1 life: Add {R}.',
             '{1}, {T}, Sacrifice Sunbaked Canyon: Draw a card.'], [0, 2]),
            ('Boros Charm', ['Target creature gains double strike until end of turn.',
             'Boros Charm deals 4 damage to target player or planeswalker.',
             'Permanents you control gain indestructible until end of turn.'], [2, 0, 1]),
            ('Arid Mesa', ['{T}, Pay 1 life, Sacrifice Arid Mesa: Search your library for a Mountain or Plains card, put it onto the battlefield, then shuffle.'], [0])]
        for name, labels, indices in cases:
            obs, bindings = fixture(name, labels)
            for option, index in zip(obs.decision.options, indices):
                self.assertEqual(option.constraints['semantic_ability'], CARD_REGISTRY[name].abilities[index].id)
                self.assertEqual(option.constraints['source'], SOURCE)
                command, args = encode_response(bindings[option.id], None)
                self.assertEqual(command, 'sendPlayerUUID')
                self.assertEqual(args['value'], option.id.removeprefix('object:'))

    def test_unknown_hidden_and_ambiguous_sources_fail_closed(self):
        for kwargs in ({'hidden': True}, {'source': None, 'second': 'Inspiring Vantage'}):
            obs, _ = fixture('Sacred Foundry', ['{T}: Add {R}.'], **kwargs)
            self.assertNotIn('semantic_ability', obs.decision.options[0].constraints)
        obs, _ = fixture('Sacred Foundry', ['{T}: Add {R}. Draw a card.'])
        self.assertNotIn('semantic_ability', obs.decision.options[0].constraints)
        obs, _ = fixture('Sacred Foundry', ['{T}: Add {R}.'], second='Inspiring Vantage')
        self.assertEqual(obs.decision.options[0].constraints['source'], SOURCE)

    def test_native_ability_id_resolves_source_without_prior_click(self):
        from patchwork_automaton.backends.burn_actions import bind_ability
        obs, _ = fixture('Sacred Foundry', ['{T}: Add {R}.'], second='Inspiring Vantage')
        view = {'canPlayObjects': {'objects': {OTHER: {'basicManaAbilities': [{'id': 'ability'}]}}}}
        bound = bind_ability('ability', '1. {T}: Add {R}.', view, obs.objects)
        self.assertEqual(bound['source'], OTHER)
        self.assertEqual(bound['semantic_card'], CARD_REGISTRY['Inspiring Vantage'].id)

    def test_uuid_cancel_remains_a_distinct_wire_response(self):
        obs, bindings = fixture('Boros Charm', ['Cancel'])
        option = obs.decision.options[0]
        self.assertEqual(option.constraints['action_kind'], 'cancel')
        self.assertEqual(encode_response(bindings[option.id], None)[0], 'sendPlayerUUID')
        self.assertEqual(encode_response(bindings['no'], None)[0], 'sendPlayerBoolean')

    def test_cast_source_survives_new_stack_uuid_without_binding_another_card(self):
        from patchwork_automaton.backends.burn_actions import bind_ability
        obs, _ = fixture('Boros Charm', [])
        obj = next(iter(obs.objects.values()))
        stack_id = Handle('xmage','object',OTHER)
        objects = {stack_id:replace(obj,id=stack_id,zone='stack')}
        source = {'id':SOURCE,'card':CARD_REGISTRY['Boros Charm'].id}
        label = '1. Boros Charm deals 4 damage to target player or planeswalker.'
        bound = bind_ability('mode',label,{},objects,source)
        self.assertEqual(bound['source'],OTHER)
        self.assertEqual(bound['semantic_ability'],901)
        self.assertEqual(bind_ability('mode',label,{},objects,{'id':SOURCE,'card':13}),{})

    def test_all_reviewed_rules_refer_to_real_abilities(self):
        from patchwork_automaton.backends.burn_actions import RULES
        for (name,label), identity in RULES.items():
            obs, _ = fixture(name,[label])
            self.assertEqual(obs.decision.options[0].constraints['semantic_ability'],identity)


@unittest.skipUnless(AVAILABLE, 'Install neural dependencies with uv')
class TensorTests(unittest.TestCase):
    def test_partial_spell_and_unique_trigger_keep_identity(self):
        from patchwork_automaton.core import Option
        from patchwork_automaton.neural.features import Encoder, candidate_row
        import numpy as np
        obs,_=fixture('Searing Blaze',[])
        obs=replace(obs,decision=replace(obs.decision,kind='GAME_SELECT',options=(Option('object:'+SOURCE,'uuid','Searing Blaze'),)),
            native={'canPlayObjects':{'objects':{SOURCE:{'basicCastAbilities':[{'id':'spell'}]}}}})
        a=CARD_REGISTRY['Searing Blaze'].abilities[0]
        np.testing.assert_array_equal(Encoder().encode(obs).arrays['candidates'][0],
            np.asarray(candidate_row('uuid','cast',coverage=a.coverage,ability=a),np.float32))
        obs,_=fixture('Eidolon of the Great Revel',[])
        obs=replace(obs,decision=replace(obs.decision,kind='GAME_TARGET',options=(Option('object:trigger','uuid','Eidolon',
            {'selection_kind':'trigger_order','source':SOURCE}),)))
        a=CARD_REGISTRY['Eidolon of the Great Revel'].abilities[1]
        np.testing.assert_array_equal(Encoder().encode(obs).arrays['candidates'][0],
            np.asarray(candidate_row('uuid','ability',coverage=a.coverage,ability=a),np.float32))

    def test_modes_colors_and_cancel_are_distinct_numeric_candidates(self):
        import numpy as np
        from patchwork_automaton.neural.features import Encoder
        for name, labels in [('Sacred Foundry', ['{T}: Add {R}.', '{T}: Add {W}.']),
            ('Boros Charm', ['Boros Charm deals 4 damage to target player or planeswalker.',
             'Permanents you control gain indestructible until end of turn.',
             'Target creature gains double strike until end of turn.', 'Cancel'])]:
            obs, _ = fixture(name, labels)
            arrays = Encoder().encode(obs).arrays
            for i in range(len(labels)):
                for j in range(i):self.assertFalse(np.array_equal(arrays['candidates'][i], arrays['candidates'][j]))
            self.assertTrue(arrays['candidate_mask'].all())
            # Once decoded, display wording and UUIDs never become model inputs.
            changed = replace(obs, decision=replace(obs.decision, options=tuple(
                replace(o, id='object:'+str(UUID(int=500+i)), label='unrelated display')
                if i<len(labels) else o for i,o in enumerate(obs.decision.options))))
            for key, value in arrays.items():np.testing.assert_array_equal(value, Encoder().encode(changed).arrays[key])
            reordered = replace(obs, decision=replace(obs.decision, options=tuple(reversed(obs.decision.options))))
            np.testing.assert_array_equal(arrays['candidates'][::-1], Encoder().encode(reordered).arrays['candidates'])

    def test_network_can_learn_to_separate_mana_colors(self):
        import numpy as np
        import tensorflow as tf
        from patchwork_automaton.neural.features import Encoder, collate
        from patchwork_automaton.neural.policy import untrained_policy
        obs, _ = fixture('Sacred Foundry', ['{T}: Add {R}.', '{T}: Add {W}.'])
        model = untrained_policy(seed=13, width=16, memory_size=16, blocks=1).model
        inputs = collate([Encoder().encode(obs).arrays])
        inputs.update(memory_in=np.zeros((1,16),np.float32),advance_memory=np.ones((1,1),np.float32))
        model(inputs)
        optimizer = tf.keras.optimizers.Adam(.01)
        @tf.function
        def step():
            with tf.GradientTape() as tape:
                logits=model(inputs)['logits']
                loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[1],logits=logits)[0]
            gradients=tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients((g,v) for g,v in zip(gradients,model.trainable_variables) if g is not None)
        for _ in range(40):step()
        self.assertGreater(float(tf.nn.softmax(model(inputs)['logits'])[0,1]),.9)
