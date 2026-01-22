"""Chess puzzles task implementation."""

import re
from typing import Any

from pipeline.tasks.base import BaseTask


class ChessPuzzlesTask(BaseTask):
    """
    Chess puzzles task from Lichess dataset.

    Goal: Given a FEN board position and a losing first move,
    find the winning response move.

    Filtered to simple 2-move tactics (one exchange).

    Supports abstention: if the model outputs <abstain>, it's tracked
    separately in metrics.
    """

    name = "chess_puzzles"
    default_template_variant = "simple"

    # Default system message
    system_message = (
        "A conversation between User and Assistant. The user presents a chess puzzle, "
        "and the Assistant solves it. The assistant first thinks about the "
        "chess position and tactics in the mind and then provides the winning move."
    )

    # Chess-specific assistant prefix (used in SFT prompts and RL runtime)
    assistant_prefix = "<think> Let me analyze this chess position and think step by step."

    # Rating buckets for stratified sampling
    RATING_BUCKETS = {
        "beginner": (400, 800),
        "intermediate": (800, 1200),
        "advanced": (1200, 1600),
        "expert": (1600, float('inf')),
    }

    def create_primitives(self, num_puzzles: int | None, seed: int = 42) -> list[dict]:
        """
        Download and filter chess puzzles from HuggingFace with stratified sampling.

        Filters to:
        - Exactly 2 moves (one exchange: losing_move + winning_move)
        - Short tactical puzzles

        Stratifies by rating buckets to get uniform distribution:
        - beginner: 400-800
        - intermediate: 800-1200
        - advanced: 1200-1600
        - expert: 1600+

        Uses streaming to avoid loading entire dataset into memory.

        Returns primitives with:
        - index: int
        - variant: str (beginner, intermediate, advanced, expert)
        - fen: str (board position)
        - losing_move: str (first move in UCI notation)
        - winning_move: str (correct response in UCI notation)
        - rating: int
        - themes: list[str]
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required for chess_puzzles task. "
                "Install with: pip install datasets"
            )

        import random

        # Calculate per-bucket target for stratified sampling
        num_buckets = len(self.RATING_BUCKETS)
        if num_puzzles is not None:
            per_bucket_target = num_puzzles // num_buckets
            # Add buffer for shuffling within each bucket
            per_bucket_collect = int(per_bucket_target * 1.5)
        else:
            per_bucket_target = float('inf')
            per_bucket_collect = float('inf')

        print("Streaming Lichess chess puzzles dataset from HuggingFace...")
        print(f"Stratified sampling by rating: {list(self.RATING_BUCKETS.keys())}")
        if num_puzzles is not None:
            print(f"Target: {per_bucket_target} per bucket, {num_puzzles} total")

        dataset = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

        # Collect puzzles into buckets
        buckets = {variant: [] for variant in self.RATING_BUCKETS}
        dataset_iter = iter(dataset)

        try:
            for i, item in enumerate(dataset_iter):
                if i % 100000 == 0 and i > 0:
                    counts = {v: len(buckets[v]) for v in buckets}
                    print(f"  Processed {i:,} puzzles, buckets: {counts}")

                # Check if all buckets are full
                all_full = all(
                    len(buckets[v]) >= per_bucket_collect
                    for v in buckets
                )
                if all_full:
                    print("All buckets full, stopping early.")
                    break

                moves = item["Moves"].strip().split()
                if len(moves) != 2:  # Only 2-move puzzles
                    continue

                rating = item["Rating"]

                # Find which bucket this rating belongs to
                variant = None
                for v, (min_r, max_r) in self.RATING_BUCKETS.items():
                    if min_r <= rating < max_r:
                        variant = v
                        break

                if variant is None:
                    continue  # Rating outside all buckets (e.g., < 400)

                # Skip if this bucket is already full
                if len(buckets[variant]) >= per_bucket_collect:
                    continue

                buckets[variant].append({
                    "fen": item["FEN"],
                    "losing_move": moves[0],
                    "winning_move": moves[1],
                    "rating": rating,
                    "themes": item["Themes"],
                    "puzzle_id": item["PuzzleId"],
                    "variant": variant,
                })
        finally:
            # Clean up the iterator to avoid threading issues
            del dataset_iter
            del dataset
            import gc
            gc.collect()

        # Report bucket sizes
        print("Bucket sizes:")
        for variant, items in buckets.items():
            min_r, max_r = self.RATING_BUCKETS[variant]
            max_str = str(int(max_r)) if max_r != float('inf') else "+"
            print(f"  {variant} ({min_r}-{max_str}): {len(items):,}")

        # Shuffle within each bucket and take target amount
        rng = random.Random(seed)
        primitives = []

        for variant in self.RATING_BUCKETS:
            bucket = buckets[variant]
            rng.shuffle(bucket)

            # Take up to per_bucket_target from each bucket
            take = min(len(bucket), per_bucket_target) if num_puzzles is not None else len(bucket)
            primitives.extend(bucket[:take])

        # Final shuffle to interleave buckets
        rng.shuffle(primitives)

        # Add indices
        for i, p in enumerate(primitives):
            p["index"] = i

        print(f"Created {len(primitives):,} primitives (stratified by rating)")
        return primitives

    def _parse_fen_to_board(self, fen: str) -> tuple[str, str]:
        """
        Parse FEN notation into human-readable board layout.

        Returns:
            (board_layout, player_who_moved_last)
        """
        # FEN format: position active_color castling en_passant halfmove fullmove
        parts = fen.split()
        position = parts[0]
        active_color = parts[1] if len(parts) > 1 else 'w'

        # Map piece notation to names
        piece_names = {
            'K': 'white king', 'Q': 'white queen', 'R': 'white rook',
            'B': 'white bishop', 'N': 'white knight', 'P': 'white pawn',
            'k': 'black king', 'q': 'black queen', 'r': 'black rook',
            'b': 'black bishop', 'n': 'black knight', 'p': 'black pawn',
        }

        # Parse board (rows are rank 8 to rank 1)
        rows = position.split('/')
        board_lines = []

        for rank_idx, row in enumerate(rows):
            rank = 8 - rank_idx  # Convert to chess rank (8, 7, 6, ..., 1)
            pieces = []
            file_idx = 0  # Column index (a-h)

            for char in row:
                if char.isdigit():
                    # Empty squares
                    file_idx += int(char)
                else:
                    # Piece
                    file = chr(ord('a') + file_idx)  # Convert to letter (a-h)
                    piece_name = piece_names.get(char, char)
                    pieces.append(f"{piece_name} {file}{rank}")
                    file_idx += 1

            if pieces:
                board_lines.append(f"Row {rank}: {', '.join(pieces)}")
            else:
                board_lines.append(f"Row {rank}: (empty)")

        board_layout = '\n'.join(board_lines)

        # Determine who is about to make the losing move
        # If active_color is 'w', white is to move (makes the losing move)
        # If active_color is 'b', black is to move (makes the losing move)
        last_mover = 'white' if active_color == 'w' else 'black'

        return board_layout, last_mover

    def _build_board_dict(self, fen: str) -> dict[str, str]:
        """
        Build a dictionary mapping square coordinates to piece characters.

        Returns:
            Dict like {'e2': 'P', 'e7': 'p', 'a1': 'R', ...}
        """
        parts = fen.split()
        position = parts[0]
        rows = position.split('/')

        board = {}
        for rank_idx, row in enumerate(rows):
            rank = 8 - rank_idx  # Convert to chess rank (8, 7, 6, ..., 1)
            file_idx = 0  # Column index (a-h)

            for char in row:
                if char.isdigit():
                    # Empty squares
                    file_idx += int(char)
                else:
                    # Piece
                    file = chr(ord('a') + file_idx)
                    square = f"{file}{rank}"
                    board[square] = char
                    file_idx += 1

        return board

    def _elaborate_uci_move(self, fen: str, uci_move: str, player: str) -> str:
        """
        Convert UCI notation to human-readable move description.

        Args:
            fen: Board position before the move
            uci_move: UCI notation (e.g., "e5d6")
            player: 'white' or 'black'

        Returns:
            Human-readable description (e.g., "black knight from e5 to d6")
        """
        # Parse UCI move
        if len(uci_move) < 4:
            return uci_move  # Invalid format, return as-is

        from_square = uci_move[:2]  # e.g., "e5"
        to_square = uci_move[2:4]   # e.g., "d6"
        promotion = uci_move[4:5] if len(uci_move) > 4 else None

        # Build board state
        board = self._build_board_dict(fen)

        # Get piece at source square
        piece_char = board.get(from_square)
        if not piece_char:
            return f"{player} moves from {from_square} to {to_square}"

        # Map piece to name (without color)
        piece_type_names = {
            'K': 'king', 'Q': 'queen', 'R': 'rook',
            'B': 'bishop', 'N': 'knight', 'P': 'pawn',
            'k': 'king', 'q': 'queen', 'r': 'rook',
            'b': 'bishop', 'n': 'knight', 'p': 'pawn',
        }
        piece_type = piece_type_names.get(piece_char, 'piece')

        # Check if destination has an opponent piece (capture)
        dest_piece = board.get(to_square)
        is_capture = dest_piece is not None

        # Format the move
        if is_capture:
            result = f"{player} {piece_type} from {from_square} captures on {to_square}"
        else:
            result = f"{player} {piece_type} from {from_square} to {to_square}"

        # Add promotion info if applicable
        if promotion:
            promo_names = {'q': 'queen', 'r': 'rook', 'b': 'bishop', 'n': 'knight'}
            promo_piece = promo_names.get(promotion.lower(), promotion)
            result += f", promoting to {promo_piece}"

        return result

    def enrich_primitive_for_rl(self, primitive: dict) -> dict:
        """
        Enrich primitive with derived fields needed for RL template substitution.

        verl's runtime template does simple string substitution, so we need to
        pre-compute fields like board_layout, other_mover, elaborated_move.

        Args:
            primitive: Raw primitive dict with fen, winning_move, losing_move, etc.

        Returns:
            Enriched primitive with additional computed fields
        """
        # Parse FEN to get board layout and determine who moved last
        board_layout, last_mover = self._parse_fen_to_board(primitive["fen"])

        # Calculate the other player (who needs to respond)
        other_mover = 'black' if last_mover == 'white' else 'white'

        # Elaborate the losing move into human-readable format
        elaborated_move = self._elaborate_uci_move(
            primitive["fen"],
            primitive["losing_move"],
            last_mover
        )

        # Return enriched primitive with all fields needed for template
        return {
            **primitive,
            "board_layout": board_layout,
            "last_mover": last_mover,
            "other_mover": other_mover,
            "elaborated_move": elaborated_move,
        }

    def format_prompt(
        self,
        primitive: dict,
        template: str,
        include_assistant_prefix: bool = True,
    ) -> list[dict]:
        """Format chess puzzle into chat messages."""
        # Parse FEN to readable format
        board_layout, last_mover = self._parse_fen_to_board(primitive["fen"])

        # Calculate the other player (who needs to respond)
        other_mover = 'black' if last_mover == 'white' else 'white'

        # Elaborate the losing move into human-readable format
        elaborated_move = self._elaborate_uci_move(
            primitive["fen"],
            primitive["losing_move"],
            last_mover
        )

        # Substitute variables in template
        content = template.replace("{{fen}}", primitive["fen"])
        content = content.replace("{fen}", primitive["fen"])
        content = content.replace("{{board_layout}}", board_layout)
        content = content.replace("{board_layout}", board_layout)
        content = content.replace("{{losing_move}}", primitive["losing_move"])
        content = content.replace("{losing_move}", primitive["losing_move"])
        content = content.replace("{{winning_move}}", primitive["winning_move"])
        content = content.replace("{winning_move}", primitive["winning_move"])
        content = content.replace("{{elaborated_move}}", elaborated_move)
        content = content.replace("{elaborated_move}", elaborated_move)
        content = content.replace("{{last_mover}}", last_mover)
        content = content.replace("{last_mover}", last_mover)
        content = content.replace("{{other_mover}}", other_mover)
        content = content.replace("{other_mover}", other_mover)

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": content},
        ]

        if include_assistant_prefix:
            messages.append({
                "role": "assistant",
                "content": self.assistant_prefix,
            })

        return messages

    def check_correctness(
        self,
        primitive: dict,
        generation: str,
    ) -> tuple[bool, dict]:
        """
        Check if the predicted move matches the winning move.

        Validates:
        1. Check for abstention (<abstain> tag)
        2. Answer can be parsed from <answer> tags
        3. Move is in valid UCI notation
        4. Move matches the winning move
        """
        # Always include expected move in metadata
        expected_move_raw = primitive["winning_move"]

        # Check for <abstain> tag first
        if "<abstain>" in generation:
            return False, {
                "predicted_move": None,
                "expected_move": expected_move_raw,
                "abstained": True,
            }

        answer = self.extract_answer(generation)

        if answer is None:
            return False, {
                "predicted_move": None,
                "expected_move": expected_move_raw,
                "error": "no_answer_tag",
                "abstained": False,
            }

        # Clean and normalize the answer (remove spaces, lowercase)
        predicted_move = answer.strip().lower().replace(" ", "")
        expected_move = expected_move_raw.strip().lower()

        # Check if predicted move is in valid UCI format (e.g., "e2e4")
        uci_pattern = r'^[a-h][1-8][a-h][1-8][qrbn]?$'
        if not re.match(uci_pattern, predicted_move):
            return False, {
                "predicted_move": answer,
                "expected_move": expected_move_raw,
                "error": "invalid_uci_format",
                "abstained": False,
            }

        # Check if move matches expected winning move
        is_correct = (predicted_move == expected_move)

        return is_correct, {
            "predicted_move": answer,
            "expected_move": expected_move_raw,
            "abstained": False,
        }

    def filter_for_sft(
        self,
        examples: list[dict],
        include_abstained: bool = True,
        include_wrong_valid_format: bool = False,
    ) -> list[dict]:
        """
        Filter examples for SFT training.

        Chess-specific filtering logic:
        - Always include correct examples
        - Optionally include abstained examples
        - Optionally include wrong answers with valid UCI format

        "Valid format" means no error field in metadata, i.e., the model
        produced a parseable UCI move (like "e2e4") even if it was wrong.
        This excludes:
        - "invalid_uci_format": answer tag content wasn't valid UCI (e.g., "Qf3c5")
        - "no_answer_tag": no <answer> tags found at all

        Args:
            examples: List of generated examples from dataset
            include_abstained: Include examples where model abstained
            include_wrong_valid_format: Include wrong answers with valid UCI format

        Returns:
            Filtered list of examples suitable for SFT training
        """
        def is_abstained(ex):
            return ex.get("abstained", False) or ex.get("metadata", {}).get("abstained", False)

        def has_valid_format_wrong(ex):
            """Check if example is wrong but has valid UCI format."""
            if ex.get("correct", False):
                return False  # Not wrong
            metadata = ex.get("metadata", {})
            # No error field means valid UCI format (even if wrong move)
            return "error" not in metadata

        filtered = []
        for ex in examples:
            if ex.get("correct", False):
                filtered.append(ex)
            elif include_abstained and is_abstained(ex):
                filtered.append(ex)
            elif include_wrong_valid_format and has_valid_format_wrong(ex):
                filtered.append(ex)

        return filtered

    def _categorize_result(self, r: dict) -> str:
        """Categorize a single result into one of: correct, abstained, incomplete, wrong."""
        is_abstained = r.get("abstained", False) or r.get("metadata", {}).get("abstained", False)

        if r.get("correct", False):
            return "correct"
        elif is_abstained:
            return "abstained"
        elif r.get("finish_reason") == "length" or r.get("error") == "no_answer_tag":
            return "incomplete"
        else:
            return "wrong"

    def compute_metrics(self, results: list[dict]) -> dict:
        """Compute metrics including four-way outcome distribution by variant."""
        from collections import defaultdict

        metrics = super().compute_metrics(results)

        # Track distribution by variant (difficulty level)
        dist_by_variant = defaultdict(lambda: {"count": 0, "correct": 0, "incomplete": 0, "abstained": 0, "wrong": 0})

        for r in results:
            variant = r.get("variant", "unknown")
            category = self._categorize_result(r)

            dist_by_variant[variant]["count"] += 1
            dist_by_variant[variant][category] += 1

        # Compute totals
        total_dist = {"count": 0, "correct": 0, "incomplete": 0, "abstained": 0, "wrong": 0}
        for v_dist in dist_by_variant.values():
            for k in total_dist:
                total_dist[k] += v_dist[k]

        metrics["distribution_by_variant"] = dict(dist_by_variant)
        metrics["distribution"] = total_dist

        # Keep legacy fields for backwards compatibility
        metrics["abstained"] = total_dist["abstained"]
        metrics["abstention_rate"] = total_dist["abstained"] / total_dist["count"] if total_dist["count"] else 0

        return metrics

    def format_metrics(self, metrics: dict, model_name: str | None = None) -> str:
        """
        Format chess puzzle metrics as a table with outcome distribution by difficulty.
        """
        lines = ["", "=== Evaluation Results ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append("")

        # Table header
        lines.append(f"{'Difficulty':<14} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Abstained':>11} {'Wrong':>8}")
        lines.append("-" * 65)

        # By variant rows (ordered by difficulty)
        dist_by_variant = metrics.get("distribution_by_variant", {})
        difficulty_order = ["beginner", "intermediate", "advanced", "expert", "unknown"]

        for variant in difficulty_order:
            if variant in dist_by_variant:
                d = dist_by_variant[variant]
                lines.append(
                    f"{variant.capitalize():<14} {d['count']:>7} {d['correct']:>10} {d['incomplete']:>12} {d['abstained']:>11} {d['wrong']:>8}"
                )

        # Total row
        lines.append("-" * 65)
        d = metrics.get("distribution", {})
        lines.append(
            f"{'Total':<14} {d.get('count', 0):>7} {d.get('correct', 0):>10} {d.get('incomplete', 0):>12} {d.get('abstained', 0):>11} {d.get('wrong', 0):>8}"
        )

        # Accuracy summary
        lines.append("")
        lines.append(f"Accuracy: {metrics.get('accuracy', 0):.1%}")

        return "\n".join(lines)

    def get_split_indices(
        self,
        total: int,
        split: str,
        seed: int = 42,
        primitives: list[dict] | None = None,
    ) -> list[int]:
        """
        Chess puzzles use custom split ratios similar to zebra_logic.

        - sft: 35%
        - rl_train: 37%
        - rl_val: 5%
        - classifier: 15%
        - eval: 8%

        Supports alternate templates (e.g., "sft_justify" uses same indices as "sft").
        """
        import random

        rng = random.Random(seed)
        indices = list(range(total))
        rng.shuffle(indices)

        splits = {
            "sft": (0.0, 0.35),
            "rl_train": (0.35, 0.72),
            "rl_val": (0.72, 0.77),
            "classifier": (0.77, 0.92),
            "eval": (0.92, 1.0),
        }

        # Check if split exists directly first
        if split in splits:
            base_split = split
        elif "_" in split:
            # Handle alternate template names (e.g., sft_justify → sft)
            base_split = split.split("_")[0]
        else:
            base_split = split

        if base_split not in splits:
            raise ValueError(f"Unknown split: {split}. Available: {list(splits.keys())} (and variants like sft_justify)")

        start_ratio, end_ratio = splits[base_split]
        start = int(total * start_ratio)
        end = int(total * end_ratio)

        return indices[start:end]
