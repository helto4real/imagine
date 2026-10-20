#!/usr/bin/env python3
"""Own CPython exception context/cause identities; no imports of Torch."""
import argparse
import json
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()

def state(errors):
    def index(value):
        return None if value is None else next(i for i, error in enumerate(errors) if error is value)
    return [dict(context=index(error.__context__), cause=index(error.__cause__),
                 suppress=error.__suppress_context__) for error in errors]

cases = []
for links in [[None,None,None], [1,2,None], [1,0,None], [0,None,None], [1,2,0]]:
    for raised in range(3):
        for current in [None,0,1,2]:
            errors = [ValueError('own ' + str(i)) for i in range(3)]
            for i, link in enumerate(links):
                errors[i].__context__ = None if link is None else errors[link]
            if current is None:
                try: raise errors[raised]
                except ValueError: pass
            else:
                try: raise errors[current]
                except ValueError:
                    try: raise errors[raised]
                    except ValueError: pass
            cases.append(dict(links=links, raised=raised, current=current, state=state(errors)))
            for error in errors:
                error.__context__ = None
                error.__traceback__ = None

properties = []
for name in ['__context__', '__cause__', '__suppress_context__']:
    for value in ['none',0,1,2,'other',True,False]:
        errors = [ValueError('own ' + str(i)) for i in range(3)]
        bound = (None if value == 'none' else (errors[value] if type(value) is int else
                 (7 if value == 'other' else value)))
        failure = None
        try: setattr(errors[0], name, bound)
        except TypeError as error: failure = [type(error).__name__, str(error)]
        properties.append(dict(name=name, value=value, state=state(errors), error=failure))
        for error in errors:
            error.__context__ = error.__cause__ = None

a.output.write_text(json.dumps(dict(schema=1, trained_model_weights=0,
    fixture_tensors=0, cases=cases, properties=properties), separators=(',', ':')) + '\n')
print('Captured',len(cases),'automatic context histories and',len(properties),'property assignments')
