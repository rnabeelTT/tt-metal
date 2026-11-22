function getTimeSinceLastSuccess(workflowName) {
  try {
    if (!__lastSuccessTimestamps) return EMPTY_VALUE;
    const info = __lastSuccessTimestamps.get(workflowName);
    if (!info) return EMPTY_VALUE;

    if (info.never_succeeded) {
      return 'Never';
    }

    if (!info.timestamp) return EMPTY_VALUE;

    const lastSuccessDate = new Date(info.timestamp);
    const now = new Date();
    const daysSince = Math.floor((now - lastSuccessDate) / (1000 * 60 * 60 * 24));

    if (daysSince === 0) return 'Today';
    if (daysSince === 1) return '1 day ago';
    return `${daysSince} days ago`;
  } catch (_) {
    return EMPTY_VALUE;
  }
}
