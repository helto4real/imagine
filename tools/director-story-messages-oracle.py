#!/usr/bin/env python3
"""Director Story message builders on synthetic browser requests.

Runs comfyui-helto-director's own server code (shared/prompt_studio.py) with
the provider call replaced by a capture: no services, settings, prompt
profiles or runtime data. Writes tests/fixtures/director-story-messages.json.
"""
import json
import subprocess
import sys
import threading
import types
from pathlib import Path

DIRECTOR = Path('/home/thhel/git/comfyui-helto-director')
FIXTURE = Path(__file__).resolve().parent.parent / 'tests/fixtures/director-story-messages.json'


class Stub(types.ModuleType):
    """ComfyUI and private packages Director imports but these builders never use."""

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return Stub(f'{self.__name__}.{name}')

    def __call__(self, *args, **kwargs):
        return None


def load():
    sys.path.insert(0, str(DIRECTOR))
    for _ in range(30):
        try:
            import shared.prompt_studio as studio
            return studio
        except ModuleNotFoundError as error:
            if error.name.startswith('shared'):
                raise
            sys.modules[error.name] = Stub(error.name)
            for name in [n for n in sys.modules if n.startswith('shared')]:
                del sys.modules[name]
    raise RuntimeError('Director imports did not settle')


class Captured(Exception):
    def __init__(self, user, images):
        super().__init__('captured')
        self.user = user
        self.images = images


def patch(studio):
    from shared.contracts.story_outline import StoryOutlineModelContract, story_outline_json_schema_text

    def system_prompt(key, *args, prompt_profile=None, story_outline_contract=None):
        if key in {'story_outline', 'story_outline_dialogue'}:
            contract = story_outline_contract or StoryOutlineModelContract(add_dialogue=key.endswith('_dialogue'))
            return story_outline_json_schema_text(contract)
        return f'SYSTEM:{key}'

    def generate(provider, settings, *, system_prompt, user_prompt, images, **kwargs):
        raise Captured(user_prompt, len(images))

    studio.system_prompt = system_prompt
    studio.generate_provider_text = generate
    studio._prompt_profile_provider_settings = lambda provider, settings, *a, **k: settings
    studio._provider_vision_support = lambda *a: None
    studio._load_visuals = lambda rows: [r for r in (rows or []) if isinstance(r, dict)]


PROFILE = {'profile': {}, 'resolved_id': 'own'}


def run(studio, kind, request):
    cancelled = studio.PromptStudioCancellationCoordinator(threading.Event())
    status = lambda *a: None
    instruction = str(request.get('user_instruction') or '').strip()
    if kind == 'outline':
        built = studio._build_story_outline(request, instruction, PROFILE, studio.StoryRequestContext())
        return {'user': built['user_prompt'], 'schema': built['system_prompt'],
                'references': [list(r) for r in built['references']]}
    try:
        if kind == 'scene':
            studio._run_story_scene(request, 'ollama', {}, instruction, status, cancelled, PROFILE)
        else:
            studio._run_image_inventory({'items': [request['item']]}, 'ollama', {}, instruction,
                                        status, cancelled, PROFILE)
    except Captured as captured:
        return {'user': captured.user, 'images': captured.images}
    raise RuntimeError(f'{kind} made no provider call')


def cast(alias, title, description, language='', voice=''):
    return {'story_names': [title, alias], 'reference_tag': f'@{alias}', 'description': description,
            'default_language': language, 'voice_guidance': voice, 'required': True}


KEEPER = cast('keeper', 'Own Keeper', '  Own fictional keeper, grey coat  ', 'Swedish', 'calm')
VISITOR = cast('visitor', 'Own Visitor', 'Own fictional visitor', 'English', 'bright')
INVENTORY = [
    {'id': 'own_setting', 'usage': 'story_reference', 'label': 'Own lighthouse', 'user_note': '',
     'inventory_note': 'A white lighthouse on rocks.', 'reuse': True},
    {'id': 'own_start', 'usage': 'story_reference', 'label': 'Own door closed', 'user_note': '',
     'inventory_note': 'A closed door "red"\nwith @keeper.', 'reuse': False},
    {'id': 'own_end', 'usage': 'story_reference', 'label': 'Own door open', 'user_note': '',
     'inventory_note': '', 'reuse': False},
    {'id': 'own_moment', 'usage': 'required_moment', 'label': 'Own beacon lit', 'user_note': '',
     'inventory_note': 'The beacon glows.', 'reuse': False},
]
BEATS = [
    {'beat_key': 'required_moment:own_moment', 'type': 'required_moment', 'order': 1,
     'image_id': 'own_moment', 'cue': ''},
    {'id': 'beat_1', 'start_image_id': 'own_start', 'end_image_id': 'own_end', 'cue': '',
     'beat_key': 'start_end_pair:beat_1', 'type': 'start_end_pair', 'order': 1},
]
CONTINUATION = {
    'schema_version': 1, 'source_scene_id': 'own_scene',
    'recent_scene_summaries': [{'title': 'Own Arrival', 'summary': '@visitor lands.'}],
    'previous_scene': {'title': 'Final Scene', 'summary': '', 'description': 'Own harbour at dusk',
                       'ambient_sound': 'Gulls', 'non_diegetic_music': '',
                       'sections': [{'prompt': '@keeper waits'}, {'prompt': 'Own keeper turns'}]},
    'context_cast': [{'reference_tag': '@keeper', 'description': 'Own fictional keeper', 'required': False}],
    'source_revision': 'own1rev',
}


def scene(dialogue=False, weight=1, section_weight=2.5):
    section = {'id': 'scene_001_section_001', 'summary': '', 'prompt': '@keeper climbs the stairs.\n@visitor follows.',
               'weight': section_weight, 'image_ids': ['own_moment'], 'first_frame_id': 'own_start',
               'last_frame_id': 'own_end'}
    if dialogue:
        section['dialogue_intent'] = {'participants': ['@keeper', '@visitor'], 'purpose': 'Own greeting',
                                      'max_turns': 2}
    return {'id': 'scene_001', 'title': 'Own Climb', 'summary': '@keeper climbs.', 'description': 'Own stairwell',
            'ambient_sound': 'Wind', 'non_diegetic_music': '', 'weight': weight, 'image_ids': ['own_setting'],
            'boundary_from_previous': 'hard_cut',
            'sections': [section, {'id': 'scene_001_section_002', 'summary': '', 'prompt': 'Own lamp turns.',
                                   'weight': 1, 'image_ids': [], 'first_frame_id': '', 'last_frame_id': ''}]}


def outline(**fields):
    request = {'operation': 'story_outline', 'user_instruction': '  Own extra instruction  ',
               'story': 'Own keeper lights the lamp.', 'cast': [], 'continuation_context': None,
               'duration_seconds': 10, 'add_dialogue': False, 'include_non_diegetic_music': False,
               'image_inventory': [], 'visual_beats': []}
    request.update(fields)
    return request


def scene_request(**fields):
    request = {'operation': 'story_scene', 'user_instruction': '', 'add_dialogue': False, 'cast': [],
               'scene': scene(), 'previous_summary': '', 'previous_scene_context': None,
               'context_cast': [], 'next_summary': '', 'images': []}
    request.update(fields)
    return request


IMAGE = {'id': 'own_moment', 'label': 'Own beacon lit', 'note': '', 'usage': 'required_moment'}

CASES = [
    ('outline_minimal', 'outline', outline()),
    ('outline_grounded_dialogue', 'outline', outline(
        story=' @keeper meets @visitor.\n@keeper waves; "Hej" — ö ', cast=[KEEPER, VISITOR], duration_seconds=20,
        add_dialogue=True, include_non_diegetic_music=True, image_inventory=INVENTORY, visual_beats=BEATS)),
    ('outline_one_cast_continued', 'outline', outline(
        story='@keeper relights the lamp.', cast=[KEEPER], continuation_context=CONTINUATION,
        duration_seconds=7.5, user_instruction='', image_inventory=INVENTORY[:1])),
    ('outline_long_short_and_odd_cast', 'outline', outline(
        story='@a.b and @keeper@x and mail@keeper', duration_seconds=0.1,
        cast=[cast('keeper', 'Anna', 'x' * 4100), cast('Keeper', 'Dup', ''),
              {'story_names': ['Bo', 'bo', 'BO', 'y' * 201], 'reference_tag': '@a.b', 'description': '',
               'required': False}, {'story_names': [], 'reference_tag': 'nope'},
              {'story_names': [], 'reference_tag': '@long_alias'}])),
    ('outline_day', 'outline', outline(duration_seconds=100000)),
    ('scene_first_no_images', 'scene', scene_request(next_summary='Own next.')),
    ('scene_dialogue_images', 'scene', scene_request(
        user_instruction='  Keep @keeper calm  ', add_dialogue=True, cast=[KEEPER, VISITOR], scene=scene(True),
        previous_summary='Own arrival.', next_summary='',
        previous_scene_context={'title': 'Own Arrival', 'summary': '@visitor lands.', 'description': '',
                                'ambient_sound': '', 'non_diegetic_music': '',
                                'sections': [{'prompt': '@visitor steps off.'}]},
        dialogue_timing={'scene_duration_seconds': 6.583333333333334,
                         'sections': [{'id': 'scene_001_section_001', 'duration_seconds': 4.5},
                                      {'id': 'scene_001_section_002', 'duration_seconds': 2.083333333333334}]},
        images=[IMAGE])),
    ('scene_dialogue_without_timing', 'scene', scene_request(
        add_dialogue=True, cast=[KEEPER, VISITOR], scene=scene(True, 2.5, 1))),
    ('scene_continued', 'scene', scene_request(
        cast=[KEEPER], previous_scene_context=CONTINUATION['previous_scene'],
        context_cast=CONTINUATION['context_cast'])),
    ('inventory_label', 'inventory', {'user_instruction': ' Own look ', 'item': IMAGE}),
    ('inventory_empty', 'inventory', {'user_instruction': '',
                                      'item': {'id': '', 'label': '', 'note': '', 'usage': ''}}),
]


def main():
    studio = load()
    patch(studio)
    commit = subprocess.check_output(['git', '-C', str(DIRECTOR), 'rev-parse', 'HEAD'], text=True).strip()
    cases = [{'name': name, 'kind': kind, 'request': request, **run(studio, kind, request)}
             for name, kind, request in CASES]
    FIXTURE.write_text(json.dumps({'source': f'comfyui-helto-director {commit} shared/prompt_studio.py',
                                   'cases': cases}, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
