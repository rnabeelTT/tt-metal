function renderCommitsTable(commits) {
  if (!Array.isArray(commits) || commits.length === 0) {
    return '<p><em>None</em></p>';
  }
  const escapeHtml = (str) => String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  const rows = commits.map(c => {
    const short = c.short || (c.sha ? c.sha.substring(0, SHA_SHORT_LENGTH) : '');
    const url = c.url || (c.sha ? `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${c.sha}` : undefined);
    const who = c.author_login ? `@${c.author_login}` : (c.author_name || 'unknown');
    const whoHtml = (c.author_login && c.author_url)
      ? `<a href="${escapeHtml(c.author_url)}">@${escapeHtml(c.author_login)}</a>`
      : escapeHtml(who);
    const shaHtml = url ? `<a href="${escapeHtml(url)}"><code>${escapeHtml(short)}</code></a>` : `<code>${escapeHtml(short)}</code>`;
    const descHtml = escapeHtml(getCommitDescription(c));
    return `<tr><td>${shaHtml}</td><td>${whoHtml}</td><td>${descHtml}</td></tr>`;
  }).join('\n');
  return [
    '<table>',
    '<thead>',
    '<tr><th>SHA</th><th>Author</th><th>Description</th></tr>',
    '</thead>',
    '<tbody>',
    rows,
    '</tbody>',
    '</table>',
    ''
  ].join('\n');
}
