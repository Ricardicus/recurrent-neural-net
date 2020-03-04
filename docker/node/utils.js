function timestampLog(log, prefix="") {
  console.log("[" + (new Date()).toISOString() + "] " + prefix + log);
}

function atob(b64Encoded) {
  try {
    var s = Buffer.from(b64Encoded, 'base64').toString();
    return s;
  } catch (e) {
    return "";
  }
}