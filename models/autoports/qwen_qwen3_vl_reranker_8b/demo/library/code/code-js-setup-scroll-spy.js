  function setupScrollSpy(items) {
    var navbarHeight = parseInt(
      getComputedStyle(document.documentElement)
        .getPropertyValue('--tt-navbar-height') || '72', 10
    );

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          /* Match on the heading node itself: two sections may legitimately
           * share a title, so the label is not a key and the id lives on the
           * section rather than on the element being observed. */
          items.forEach(function (item) {
            item.link.classList.toggle('active', item.heading === entry.target);
          });
        }
      });
    }, {
      rootMargin: '-' + (navbarHeight + 8) + 'px 0px -70% 0px',
      threshold: 0
    });

    items.forEach(function (item) {
      observer.observe(item.heading);
    });

    items[0].link.classList.add('active');
  }
