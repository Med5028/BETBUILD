const CACHE = "wettrechner-v1";
const DATEIEN = ["./", "./index.html", "./manifest.json",
                 "./icon-192.png", "./icon-512.png", "./icon-maskable.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(DATEIEN)));
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(k =>
    Promise.all(k.filter(n => n !== CACHE).map(n => caches.delete(n)))));
  self.clients.claim();
});

// Netz zuerst, damit neue Daten sofort ankommen; offline aus dem Cache
self.addEventListener("fetch", e => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    fetch(e.request)
      .then(r => {
        const kopie = r.clone();
        caches.open(CACHE).then(c => c.put(e.request, kopie));
        return r;
      })
      .catch(() => caches.match(e.request).then(t => t || caches.match("./index.html")))
  );
});
