const DB_NAME = "vizmlir";
const DB_VERSION = 1;
const SESSIONS = "sessions";
const KV = "kv";

let dbPromise = null;

function openDb() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = () => {
        const db = request.result;

        if (!db.objectStoreNames.contains(SESSIONS)) {
          db.createObjectStore(SESSIONS, { keyPath: "id" });
        }

        if (!db.objectStoreNames.contains(KV)) db.createObjectStore(KV);
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  return dbPromise;
}

async function run(storeName, mode, action, fallback) {
  try {
    const db = await openDb();

    return await new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, mode);
      const request = action(tx.objectStore(storeName));
      tx.oncomplete = () => resolve(request?.result);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } catch (error) {
    console.warn("vizmlir storage unavailable:", error);

    return fallback;
  }
}

export const kv = {
  get: (key) => run(KV, "readonly", (store) => store.get(key), null),
  set: (key, value) =>
    run(KV, "readwrite", (store) => store.put(value, key), null),
  delete: (key) => run(KV, "readwrite", (store) => store.delete(key), null),
};

export const sessions = {
  async list() {
    const all = await run(SESSIONS, "readonly", (s) => s.getAll(), []);

    return (all ?? []).sort((a, b) => b.savedAt - a.savedAt);
  },
  get: (id) => run(SESSIONS, "readonly", (store) => store.get(id), null),
  put: (session) =>
    run(SESSIONS, "readwrite", (store) => store.put(session), null),
  delete: (id) => run(SESSIONS, "readwrite", (store) => store.delete(id), null),
};

export const SESSION_FORMAT = "vizmlir-session";

export function sessionToFile(name, state) {
  return JSON.stringify(
    { format: SESSION_FORMAT, version: 1, name, savedAt: Date.now(), state },
    null,
    2,
  );
}

export function sessionFromFile(text) {
  let data;

  try {
    data = JSON.parse(text);
  } catch {
    return null;
  }

  if (data?.format !== SESSION_FORMAT || typeof data.state !== "object") {
    return null;
  }

  return { name: String(data.name ?? "session"), state: data.state };
}
