# v4 sample tiles

Pictures only. The `.npz` files are not in this repo.

Each file is one 512 tile. Left is HH. Right is the v4 label: copper is layover, blue is shadow, the hatch is ignore (255). The sensor is at the top, so layover sits on the near side of a roof and shadow on the far side.

These six were picked because a building is in the frame. They are not a draw from the class prior. On the train split, layover is about 5% of the pixels that are labelled.

| file | what | strip | split | ignored | layover | shadow |
|---|---|---|---|---:|---:|---:|
| [01_port_block.png](01_port_block.png) | port, a block | `…22074521` | train | 15% | 10% | 3% |
| [02_port_ignore.png](02_port_ignore.png) | port, clutter left unlabelled | `…22074521` | train | 43% | 44% | 4% |
| [03_kop.png](03_kop.png) | Kop van Zuid | `…23072925` | train | 34% | 63% | 9% |
| [04_north.png](04_north.png) | north look | `…22071321` | train | 28% | 64% | 5% |
| [05_val.png](05_val.png) | val | `…04144102` | val | 8% | 65% | 10% |
| [06_test.png](06_test.png) | test | `…22133635` | test | 17% | 53% | 13% |

Layover and shadow are shares of the pixels that are not ignored. The port pair is the clutter point: on 02 the bright lines that are not buildings are hatched, not painted as layover.
