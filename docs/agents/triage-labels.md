# Triage roles

Use these roles in local incoming-request files as `Triage: <role>`. They are separate from ticket execution status. Tickets produced by an approved split can use `ready-for-agent` directly. If a request is explicitly tracked on GitHub, use the same names as labels.

| Label in mattpocock/skills | Local role / GitHub label | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill asks to apply a triage label, update the local `Triage` field by default.
