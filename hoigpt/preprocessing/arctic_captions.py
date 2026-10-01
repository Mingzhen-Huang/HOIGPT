"""Recreate a legacy ARCTIC caption from locally obtained descriptions.

The published clip map stores only a caption digest. ARCTIC description text
is read from the user's own copy of the dataset.
"""

from pathlib import Path


def caption_candidates(descriptions_root, source, start, end):
    root = Path(descriptions_root).resolve()
    relative = Path(source)
    path = (root / relative / "description.txt").resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"ARCTIC description path leaves the dataset root: {source}")
    parsed = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        begin, finish = map(int, parts[0].split("-"))
        action = " ".join(parts[1:-2])[:-1]
        hand = " ".join(parts[-2:])
        if begin in parsed:
            if "both" in hand:
                parsed[begin] = (begin, finish, [action], ["both hand"])
        else:
            parsed[begin] = (begin, finish, [action], [hand])

    entries = sorted(parsed.values(), key=lambda row: row[0])
    if not entries:
        return []
    groups = [[entries[0]]]
    for entry in entries[1:]:
        if entry[0] <= groups[-1][-1][1] + 5:
            groups[-1].append(entry)
        else:
            groups.append([entry])

    obj = Path(source).name.split("_", 1)[0]
    matches = []
    for group in groups:
        augmented = []
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 2, len(group) + 1):
                    subset = group[i:j]
                    augmented.append((
                        subset[0][0], subset[-1][1],
                        sum((row[2] for row in subset), []),
                        sum((row[3] for row in subset), []),
                    ))
        augmented.extend(group)
        for begin, finish, actions, hands in augmented:
            if (min(begin, finish), max(begin, finish)) != (start, end):
                continue
            matches.append(", then ".join(
                " ".join((action, obj, "with", hand))
                for action, hand in zip(actions, hands)
            ))
    return sorted(set(matches))
