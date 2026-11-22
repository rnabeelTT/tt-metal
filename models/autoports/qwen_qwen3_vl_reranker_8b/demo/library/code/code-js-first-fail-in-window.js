function findFirstFailInWindow(mainBranchRunsWindow) {
  let seenAnyFailure = false;
  let firstFailInStreak = null; // oldest failure observed before crossing a success boundary

  for (const run of mainBranchRunsWindow) {
    if (run.conclusion === 'success') {
      if (firstFailInStreak) {
        // We found a success after observing failures: return the oldest failure in the streak
        return { run: firstFailInStreak, boundarySuccessRun: run, noSuccessInWindow: false };
      }
      // Success encountered before any failure in the current scan; keep scanning older entries
    } else if (run.conclusion && run.conclusion !== 'cancelled' && run.conclusion !== 'skipped') {
      // Treat anything non-success, non-cancelled/skipped as failure for this purpose
      seenAnyFailure = true;
      firstFailInStreak = run; // update to become oldest failure within the current failing streak (and window)
    }
  }

  if (seenAnyFailure) {
    // No success found in-window; report oldest failure in the window
    return { run: firstFailInStreak, boundarySuccessRun: undefined, noSuccessInWindow: true };
  }
  return null;
}
