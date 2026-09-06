"""Choose an existing person for first CAM++ enrollment from independent evidence.

Never guess from engagement, a camera merely seeing a face, or the closest
voice profile. Callers supply actual interval mouth-motion observations or an
explicit self-identification matched to an existing person.
"""

def target(*, observations, windows, explicit_person_id=None):
    visual_ids = {r.get('person_db_id') for r in observations if r.get('person_db_id') is not None}
    window_ids = {r.get('person_id') for r in windows if r.get('person_id') is not None}
    if len(visual_ids)>1 or len(window_ids)>1 or any(r.get('change_suspected') for r in windows):
        return None
    visual = next(iter(visual_ids), None)
    if explicit_person_id is not None:
        if visual is not None and visual != explicit_person_id:
            return None
        return explicit_person_id
    rows = [r for r in observations if r.get('person_db_id') == visual
            and float(r.get('confidence') or 0) >= .5]
    if visual is not None and len(rows)>=3:
        return visual
    return None
