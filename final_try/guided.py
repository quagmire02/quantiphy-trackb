"""Question-local prompts and joint grounding of that question's own objects."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.ground import parse_boxes, resolve_conflicts


def context(row):
    return (f'Current question: {row.question}\nGiven prior: {row.prior}\n'
            f'Camera-distance annotations for this question: {row.depth_info}\n'
            'Use only this question and its supplied information. Preserve color, position and instance distinctions. '
            'Localize the actual physical object, not text labels, its shadow, or the entire background. '
            'A size question needs the whole named object, not just the dimension edge.')


class ContextVLM:
    def __init__(self, vlm, row):
        self.vlm, self.row = vlm, row

    def chat(self, messages, **kwargs):
        messages = copy.deepcopy(messages)
        if messages and messages[0].get('role') == 'system':
            messages[0]['content'].append({'type': 'text', 'text': context(self.row)})
        else:
            messages.insert(0, {'role': 'system', 'content': [{'type': 'text', 'text': context(self.row)}]})
        return self.vlm.chat(messages, **kwargs)


def decode_objects(reply, wanted):
    """Accept only exact requested object IDs and normalized boxes."""
    for start, char in enumerate(reply):
        if char not in '[{':
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(reply[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            value = value.get('objects', [])
        if not isinstance(value, list):
            continue
        result = {}
        for rec in value:
            if not isinstance(rec, dict) or not isinstance(rec.get('object_id'), str) or rec.get('object_id') not in wanted:
                continue
            key = rec['object_id']
            if key in result:
                return {}  # Duplicate IDs are ambiguous; do not silently choose one.
            boxes = parse_boxes(json.dumps(rec.get('boxes', [])))
            result[key] = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
        if result:
            return result
    return {}


def ground_one(engine, df, parsed, cache, vlm, proposal_cache=None):
    """Jointly locate reference and target(s) from exactly ONE test question."""
    if len(df) != 1:
        raise ValueError('Grounding scope must contain exactly one question')
    row = next(df.itertuples())
    path = cache / 'ground' / f'{row.video_id}.json'
    if path.exists():
        return
    rec = parsed.get(row.id)
    if not rec or not rec['ok']:
        return
    frames = engine.read_frames(row.video_path, engine.TRACK_LONGEST)
    n, h, w = frames.shape[:3]
    plan = engine.key_frames(df, parsed, n, float(row.fps))
    _, target_keys = engine.object_keys(rec)
    needed = {key: sum(t.split('#')[0] == key for t in target_keys) for key in plan}
    needed = {key: max(1, count) for key, count in needed.items()}
    object_ids = {f'o{i}': key for i, key in enumerate(plan)}
    wanted = [{'object_id': oid, 'phrase': plan[key][1], 'instances': needed[key]} for oid, key in object_ids.items()]
    prompt = (context(row) + '\nLocate these objects TOGETHER in this frame so identities are consistent: '
              + json.dumps(wanted) + '\nReturn JSON only: [{"object_id":"o0","boxes":[[x1,y1,x2,y2]]}]. '
              'Coordinates normalized to 0..1000. Return [] boxes if not visible. For two instances return two distinct boxes. '
              'Do not merge distinct colored or positioned objects. Include the complete silhouette.')
    if proposal_cache:
        proposal_path = proposal_cache / 'items' / str(int(row.id)) / 'ground' / f'{row.video_id}.json'
        if proposal_path.exists():
            helper = json.loads(proposal_path.read_text())
            if helper.get('question_id') != int(row.id):
                raise ValueError('Helper grounding belongs to another question')
            shape = helper.get('shape', [])
            if shape == [h, w]:
                candidates = [{'phrase': obj.get('phrase', key), 'frame': obj.get('frame'),
                               'boxes_0_to_1000': [[b[0]*1000/w, b[1]*1000/h, b[2]*1000/w, b[3]*1000/h] for b in obj.get('boxes', [])]}
                              for key, obj in helper.get('objects', {}).items()]
                prompt += ('\nUnverified 4B object proposals for THIS question, with original frame indices: ' + json.dumps(candidates)
                           + '\nUse them only as hypotheses. A proposal from a different frame may have moved. '
                           'Re-identify every object in the current image and correct wrong identities/boxes.')
    objects = {key: {'phrase': plan[key][1], 'frame': None, 'boxes': [], 'tried': []} for key in plan}
    order = list(dict.fromkeys([v[0] for v in plan.values()] + [n // 2, n // 4, 3*n // 4]
                              + np.linspace(0, n-1, engine.GROUND_SWEEP_FRAMES).round().astype(int).tolist()))
    for fi in order:
        current_prompt = prompt + f'\nCurrent original frame index: {fi}.'
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': current_prompt}]}]
        image = Image.fromarray(frames[fi])
        reply = vlm.chat(messages, images=[image], max_new_tokens=512)
        proposals = decode_objects(reply, object_ids)
        # Verification sees the same question/frame and unverified candidate boxes only.
        if any(proposals.values()):
            verify = current_prompt + '\nCheck these candidate boxes against the IMAGE. Correct mistaken identities, partial-object boxes '
            verify += 'and overlapping boxes for distinct objects. Return the same JSON schema with verified/corrected boxes: ' + json.dumps(proposals)
            checked = vlm.chat([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': verify}]}],
                               images=[image], max_new_tokens=512)
            proposals = decode_objects(checked, object_ids)
        else:
            checked = ''
        for oid, key in object_ids.items():
            if objects[key]['boxes']:
                continue
            objects[key]['tried'].append({'frame': int(fi), 'reply': reply, 'verification': checked})
            boxes = proposals.get(oid, [])
            if len(boxes) != needed[key]:
                continue
            pixel = [[round(x1*w/1000, 1), round(y1*h/1000, 1), round(x2*w/1000, 1), round(y2*h/1000, 1)]
                     for x1, y1, x2, y2 in boxes]
            pixel = [b for b in pixel if b[2]-b[0] >= 2 and b[3]-b[1] >= 2]
            if len(pixel) >= needed[key]:
                objects[key].update(frame=int(fi), boxes=pixel, found_by='joint_question_verified')
        if all(o['boxes'] for o in objects.values()):
            break
    objects = resolve_conflicts(ContextVLM(vlm, row), frames, objects)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'question_id': int(row.id), 'shape': [h, w], 'n_frames': n, 'objects': objects}))


def scopes(df, cache):
    """Separate caches even when questions share a video; never share semantic artifacts."""
    for i in range(len(df)):
        one = df.iloc[[i]].copy()
        yield one, cache / 'items' / str(int(one.id.iloc[0]))
