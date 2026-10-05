"""`python -m pipeline.tasks.leetcode.verifier`: run the verifier service.

The entry point lives here rather than in service.py because importing the
package's parent (pipeline.tasks.leetcode) already imports service, so
`-m ...service` would execute that module a second time as __main__.
"""

from pipeline.tasks.leetcode.verifier.service import main

main()
