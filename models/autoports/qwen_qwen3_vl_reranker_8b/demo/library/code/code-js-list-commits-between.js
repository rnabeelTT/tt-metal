function listCommitsBetweenOffline(context, startShaExclusive, endShaInclusive) {
  if (!Array.isArray(__commitsIndex) || __commitsIndex.length === 0) return [];
  const commits = __commitsIndex;
  const startIdx = commits.findIndex(c => c.sha === startShaExclusive);
  const endIdx = commits.findIndex(c => c.sha === endShaInclusive);
  if (endIdx === -1) return [];
  // We want commits strictly after start and up to and including end
  const from = startIdx === -1 ? 0 : (startIdx + 1);
  const to = endIdx;
  const slice = commits.slice(from, to + 1);
  return slice.map(c => ({
    sha: c.sha,
    short: (c.short || (c.sha ? c.sha.substring(0, SHA_SHORT_LENGTH) : '')),
    url: c.url || (c.sha ? `https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${c.sha}` : ''),
    author_login: c.author_login,
    author_name: c.author_name,
    author_url: c.author_url,
    message: c.message,
    description: getCommitDescription(c),
  }));
}
