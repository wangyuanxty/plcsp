"""v5: E2E-MAPPO base for multi-target FJSP (MT-FJSP).

Provenance
----------
Upstream : https://github.com/RKWin93/E2E-MAPPO-for-MT-FJSP.git
Commit   : bdd8719144a62f66c6c16c4243056047c050dc6d
License  : MIT (Copyright (c) 2024 RKWin) -- see ``LICENSE``
Paper    : R. Wang, Y. Jing, C. Gu, S. He, J. Chen,
           "End-to-End Multitarget Flexible Job Shop Scheduling With Deep
           Reinforcement Learning", IEEE Internet of Things Journal,
           vol. 12, no. 4, pp. 4420-4434, 2025. doi:10.1109/JIOT.2024.3485748

Local modifications are listed in ``NOTICE.md``. Search for ``[local fix]``
in the source. Absolute paths were rewritten from the upstream checkout to
``plcsp_v5``.

Entry points
------------
Training : ``python plcsp_v5/Run.py``
Testing  : ``python plcsp_v5/test_all.py``

The code uses absolute imports (``import parameters``, ``from algorithm ...``),
so run it with ``plcsp_v5`` as the working directory or on ``sys.path``.
"""
