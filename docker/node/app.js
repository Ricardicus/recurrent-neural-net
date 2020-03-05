
const http = require('http');
var fs = require('fs');
const fork = require('child_process').fork;
const path = require('path');

// file is included here:
eval(fs.readFileSync('utils.js')+'');

// Change to what is preferred. Im just running locally!
const hostname = '0.0.0.0';
const port = 8080;

// Task mgr
const taskMgr = path.resolve('taskmgr.js');
const taskMgrParams = []
const options = {
  stdio: ['pipe', 'pipe', 'pipe', 'ipc']
};

const taskMgrProc = fork(taskMgr, taskMgrParams, options);

// use event hooks to provide a callback to execute when data are available: 
taskMgrProc.stdout.on('data', function(data) {
    console.log(data.toString());
});

taskMgrProc.on('close', (code) => {
  console.log(`taskMgrProc process exited with code ${code}`);
});

// To keep track of latest output
var lastOutput = "";
var lastOutputMonitor = "";

taskMgrProc.on('message', message => {

  if ( message.startsWith("output:") ) {
    lastOutput = message.substring(7);
  } else if ( "output-monitor:" ) {
    lastOutputMonitor = message.substring(15);
  }

});

// To keep track of latest command
var lastCommand = "";

// Continously ask for output from subprocess
var askForSubProcessOutput = false;
var askForSubProcessOutputMonitor = false;

setInterval(function() {
  if ( askForSubProcessOutput ) {
    taskMgrProc.send("output");
  }
  if ( askForSubProcessOutputMonitor ) {
    taskMgrProc.send("output-monitor");
  }
}, 1000);

// URL port aliases
var aliases = {"/": "/index.html"};

function not_found(res) {
  res.statusCode = 404;
    res.end();
}

function redirect(res, dest) {
  res.writeHead(302, {
    'Location': dest
  });
  res.end();
}

function ok(res) {
  res.statusCode = 200;
  res.end();
}

function sendJson(response, object) {
  response.writeHead(200, {"Content-Type": "application/json"});
  var json = JSON.stringify(object);
  response.end(json);
}

function month_name_from_string_nbr(string_nbr)
{
  if (string_nbr == "01")
    return "Jan";
  if (string_nbr == "02")
    return "Feb";
  if (string_nbr == "03")
    return "Mar";
  if (string_nbr == "04")
    return "Apr";
  if (string_nbr == "05")
    return "May";
  if (string_nbr == "06")
    return "Jun";
  if (string_nbr == "07")
    return "Jul";
  if (string_nbr == "08")
    return "Aug";
  if (string_nbr == "09")
    return "Sep";
  if (string_nbr == "10")
    return "Oct";
  if (string_nbr == "11")
    return "Nov";
  if (string_nbr == "12")
    return "Dec";
}

function outputFile(res, url) {
  try {

    var stats = fs.lstatSync(url);

    if ( stats.isFile() ) {

      var html = fs.readFileSync(url);

      // mime type handling
      var datatype = "text/plain";

      if ( url.endsWith(".csv") ) {
        datatype = "text/csv";
      } else if ( url.endsWith(".jpg") ) {
        datatype = "image/jpeg";
      } else if ( url.endsWith(".ico") || url.endsWith(".png") ) {
        datatype = "image/png";
      } else if ( url.endsWith(".html")) {
        datatype = "text/html";
      }

      res.writeHead(200, {'Content-Type': datatype});
      res.end(html);

      return true;

    }

  } catch (e) {};

  return false;
}

const server = http.createServer((req, res) => {

  if ( req.url in aliases ) {
    req.url = aliases[req.url];
  }

  var url = req.url.replace(/\/+/g,'/').substr(1);

  var ip = req.headers['x-forwarded-for'] || 
    req.connection.remoteAddress || 
    req.socket.remoteAddress ||
    (req.connection.socket ? req.connection.socket.remoteAddress : null);

  if ( url.startsWith("launch.cgi?") ) {
    var base64Start = url.split(".cgi?")[1];

    lastCommand = atob(base64Start);
    var startCommand = "start:"+lastCommand;

    taskMgrProc.send(startCommand);

    askForSubProcessOutput = true;

    ok(res);
  } else if ( url.startsWith("launchMonitor.cgi?") ) {
    var base64Start = url.split(".cgi?")[1];

    var startCommand = "monitor-start:"+atob(base64Start);

    taskMgrProc.send(startCommand);

    askForSubProcessOutputMonitor = true;

    ok(res);
  } else if ( url.startsWith("getOutput.cgi") ) {
    var output = {"output": lastOutput};
    sendJson(res, output);
  } else if ( url.startsWith("getMonitorOutput.cgi") ) {
    var output = {"output": lastOutputMonitor};
    sendJson(res, output);
  } else if ( url.startsWith("getLastCommand.cgi") ) {
    var output = {"output": lastCommand};
    sendJson(res, output);
  } else {
    /* Not a CGI call */
    timestampLog( ip + ": " + url);

    var filereq = outputFile(res,url);
    if ( ! filereq ) {
      redirect(res, "/");
    }
  }

});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
