# Local patches to the built Studio bundle

The Studio UI is built in a separate repository; this directory holds only its
compiled output. Three fixes below were applied **directly to the built assets**
because the defects are user-visible in every deployment. Each needs the
equivalent change upstream, after which these notes can go.

Until then, note that a frontend rebuild silently reverts all three.

---

## 1. `connectionStatus` never reached `"connected"`

**File:** `static/js/main.8ca7b978.js`

`ConnectionSetup.handleConnect` calls `setIsConnected(true)` but never
`setConnectionStatus('connected')`. The header renders `connectionStatus`,
which starts at `'disconnected'`, so after a *successful* connect the UI
showed "Connected" in one indicator and "Disconnected" — with a Retry
button — in another, while every backend call was returning 200.
`connectionStatus` only became `'connected'` via `attemptReconnection()`,
i.e. only after a failure and recovery.

The patch makes the store setter keep both flags in agreement, since they
must never disagree:

```js
// before
setIsConnected:t=>e({isConnected:t})
// after
setIsConnected:t=>e({isConnected:t,connectionStatus:t?"connected":"disconnected"})
```

**Upstream fix** — in `store/useStudioStore.ts`:

```ts
setIsConnected: (isConnected) =>
  set({ isConnected, connectionStatus: isConnected ? 'connected' : 'disconnected' }),
```

## 2. Google Fonts were fetched at page load

**File:** `static/css/main.961204dc.css`

The stylesheet opened with two `@import url(https://fonts.googleapis.com/...)`
rules for Inter and JetBrains Mono. A self-hosted admin UI should not fetch
assets from a third party at page load: it breaks in air-gapped or
egress-restricted deployments (where the request hangs or resets before the
page paints) and it sends every viewer's IP and User-Agent to Google, which
is a compliance question for enterprise operators.

Both `@import` lines were removed. Every `font-family` in the bundle already
declared a full fallback stack (system UI font, then a standard monospace),
so the UI renders natively with no external request.

**Upstream fix**: drop the imports and either accept the system stack or
self-host the WOFF2 files with a local `@font-face`.

## 3. No favicon was declared

**File:** `index.html`

Nothing declared an icon, so every browser fell back to `/favicon.ico` at the
origin root — a path the platform does not serve. That produced a 404 in
every deployment's access log and a console error for every Studio user.

An inline `data:image/svg+xml` icon was added before `<title>`, which costs no
request at all. `imgs/logo.png` was not used: it is a 588 KB JPEG (despite the
extension) at 1024x1024, far too heavy for a tab icon.

**Upstream fix**: declare an icon in `public/index.html`, inline or as a small
self-hosted `.svg`/`.ico`.

---

## Source maps

`main.8ca7b978.js.map` was **not** regenerated after patch 1. The added
characters shift every column mapping that follows on that line, so
positions in the map are approximate from that point on. Regenerate the
bundle upstream rather than trusting the map around the store definition.
