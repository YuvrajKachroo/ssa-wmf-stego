"""Phase 10: extraction robustness demonstration.

Demonstrates:
1. Correct-key extraction
2. Wrong-key behavior
3. Lossless disk round-trip
4. Single-pixel tamper behavior
"""

from pathlib import Path

import cv2
import numpy as np

from src.embedding.extraction import (
    StegoKey,
    embed_secret,
    extract_secret,
)


OUTPUT_DIR = Path("outputs") / "phase10"


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Deterministic cover
    # ---------------------------------------------------------
    rng = np.random.default_rng(123)

    image = rng.integers(
        0,
        256,
        size=(64, 64),
        dtype=np.uint8,
    )

    cover_flat = image.flatten()

    # w = 2, so 4096 pixels are already divisible by 2.
    n = len(cover_flat)

    # Independent directional costs for demonstration.
    cost_rng = np.random.default_rng(1)

    rho_plus = cost_rng.uniform(
        0.1,
        10.0,
        n,
    )

    rho_minus = cost_rng.uniform(
        0.1,
        10.0,
        n,
    )

    key = StegoKey(
        h=5,
        w=2,
        seed=99,
    )

    secret = b"Phase 10 extraction demonstration."

    # ---------------------------------------------------------
    # Embed
    # ---------------------------------------------------------
    result = embed_secret(
        cover_flat,
        rho_plus,
        rho_minus,
        secret,
        key,
    )

    stego = result.stego.reshape(image.shape)

    cv2.imwrite(
        str(OUTPUT_DIR / "cover.png"),
        image,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "stego.png"),
        stego,
    )

    # ---------------------------------------------------------
    # Correct key
    # ---------------------------------------------------------
    print("--- Correct key ---")

    recovered = extract_secret(
        result.stego,
        key,
    )

    print(recovered)

    assert recovered == secret

    # ---------------------------------------------------------
    # Wrong key
    # ---------------------------------------------------------
    print("--- Wrong key (seed+1) ---")

    wrong_key = StegoKey(
        h=key.h,
        w=key.w,
        seed=key.seed + 1,
    )

    try:
        recovered_wrong = extract_secret(
            result.stego,
            wrong_key,
        )

        print(
            "Recovered:",
            recovered_wrong,
        )

        assert recovered_wrong != secret

    except ValueError as exc:
        print(
            "ValueError:",
            exc,
        )

    # ---------------------------------------------------------
    # Disk round-trip
    # ---------------------------------------------------------
    print("--- Disk round-trip ---")

    path = OUTPUT_DIR / "stego_roundtrip.pgm"

    written = cv2.imwrite(
        str(path),
        stego,
    )

    assert written

    reloaded = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    assert reloaded is not None

    assert np.array_equal(
        reloaded,
        stego,
    )

    recovered_disk = extract_secret(
        reloaded.flatten(),
        key,
    )

    print(
        "Recovered:",
        recovered_disk,
    )

    assert recovered_disk == secret

    # ---------------------------------------------------------
    # Tampering
    # ---------------------------------------------------------
    print("--- Tampered (pixel 0 shifted by 1) ---")

    tampered = result.stego.copy()

    if tampered[0] < 255:
        tampered[0] += 1
    else:
        tampered[0] -= 1

    try:
        recovered_tampered = extract_secret(
            tampered,
            key,
        )

        print(
            "Recovered after tamper:",
            recovered_tampered,
        )

        # Tampering is not guaranteed to destroy every byte,
        # but the extracted payload must not silently be treated
        # as guaranteed authentic data.
        if recovered_tampered == secret:
            print(
                "Tamper did not alter the decoded payload "
                "for this particular pixel/trellis position."
            )
        else:
            print("Tamper changed the extracted payload.")

    except ValueError as exc:
        print(
            "ValueError after tamper:",
            exc,
        )

    print()
    print("========== PHASE 10 RESULT ==========")
    print("Correct-key extraction : PASS")
    print("Disk round-trip        : PASS")
    print("Wrong-key test         : OBSERVED")
    print("Tamper test            : OBSERVED")
    print("=====================================")


if __name__ == "__main__":
    main()
