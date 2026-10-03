"""Compatibility exports for the shared, pure edit and diff mechanisms."""

from omh._tool_utils.edit_diff import (
    AppliedEditsResult as AppliedEditsResult,
)
from omh._tool_utils.edit_diff import (
    Edit as Edit,
)
from omh._tool_utils.edit_diff import (
    FuzzyMatchResult as FuzzyMatchResult,
)
from omh._tool_utils.edit_diff import (
    Replacement as Replacement,
)
from omh._tool_utils.edit_diff import (
    apply_edits_to_normalized_content as apply_edits_to_normalized_content,
)
from omh._tool_utils.edit_diff import (
    apply_replacements_preserving_unchanged_lines as apply_replacements_preserving_unchanged_lines,
)
from omh._tool_utils.edit_diff import (
    detect_line_ending as detect_line_ending,
)
from omh._tool_utils.edit_diff import (
    fuzzy_find_text as fuzzy_find_text,
)
from omh._tool_utils.edit_diff import (
    generate_diff_string as generate_diff_string,
)
from omh._tool_utils.edit_diff import (
    generate_unified_patch as generate_unified_patch,
)
from omh._tool_utils.edit_diff import (
    normalize_for_fuzzy_match as normalize_for_fuzzy_match,
)
from omh._tool_utils.edit_diff import (
    normalize_to_lf as normalize_to_lf,
)
from omh._tool_utils.edit_diff import (
    restore_line_endings as restore_line_endings,
)
from omh._tool_utils.edit_diff import (
    strip_bom as strip_bom,
)
