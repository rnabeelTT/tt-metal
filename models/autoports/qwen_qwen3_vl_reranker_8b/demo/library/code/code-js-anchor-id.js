  function anchorId(h) {
    if (h.id) return h.id;

    var node = h.parentNode;
    while (node && node.nodeType === 1) {
      if (node.tagName === 'SECTION' || node.classList.contains('section')) {
        if (node.id) return node.id;
      }
      node = node.parentNode;
    }

    /* Sphinx's own permalink already points at the right anchor. */
    var headerlink = h.querySelector('a.headerlink');
    var href = headerlink && headerlink.getAttribute('href');
    if (href && href.charAt(0) === '#') return decodeURIComponent(href.slice(1));

    return '';
  }
