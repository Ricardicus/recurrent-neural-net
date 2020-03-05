const { spawn } = require('child_process');

var trainerProc;
var trainerProcValid = false;

var monitorProc;
var monitorProcValid = false;
var fs = require('fs');

var outputLastProc = "";
var outputLastMonitorProc = "";

// file is included here:
eval(fs.readFileSync('utils.js')+'');

function taskMgrLog(message) {
  timestampLog(message, "TaskManager: ");
}

process.on('message', message => {

  if ( message.startsWith("start:") ) {
    var launchString = message.substring(6);
    var whatToLaunch = launchString.split(" ");

    var exe = whatToLaunch[0]
    var args = whatToLaunch.slice(1);

    try {

      /* Stop previous process if active */
      if ( trainerProcValid ) {

        taskMgrLog("Will stop child process");

        try {
          trainerProc.kill("SIGTERM");
        } catch ( e ) {
          taskMgrLog("Failed to send 'kill'");
        }

      }

      taskMgrLog("Starting process: " + launchString);

      /* Start new process */
      trainerProc = spawn(exe, args)
      trainerProcValid = true;

      trainerProc.stdout.on('data', (data) => {
        outputLastProc = data;
      });

      trainerProc.stderr.on('data', (data) => {
        outputLastProc = data;
      });

      trainerProc.on('close', (code) => {
        taskMgrLog(`child process exited with code ${code}`);
        trainerProcValid = false;
      });

      trainerProc.on('error', (code) => {
        taskMgrLog(`child process exited with code ${code}, bad command`);
        trainerProcValid = false;
      });

    } catch ( e ) {
      taskMgrLog("Failed to start " + exe);
    }

  } else if ( message.startsWith("monitor-start:") ) {

    var launchString = message.substring(14);
    var whatToLaunch = launchString.split(" ");

    var exe = whatToLaunch[0]
    var args = whatToLaunch.slice(1);

    try {

      /* Stop previous process if active */
      if ( monitorProcValid ) {

        taskMgrLog("Will stop child process");

        try {
          monitorProc.kill("SIGTERM");
        } catch ( e ) {
          taskMgrLog("Failed to send 'kill'");
        }

      }

      taskMgrLog("Starting process: " + launchString);

      /* Start new process */
      monitorProc = spawn(exe, args)
      monitorProcValid = true;

      monitorProc.stdout.on('data', (data) => {
        outputLastMonitorProc = data;
      });

      monitorProc.stderr.on('data', (data) => {
        outputLastMonitorProc = data;
      });

      monitorProc.on('close', (code) => {
        taskMgrLog(`child process exited with code ${code}`);
        monitorProcValid = false;
      });

      monitorProc.on('error', (code) => {
        taskMgrLog(`child process exited with code ${code}, bad command`);
        monitorProcValid = false;
      });

    } catch ( e ) {
      taskMgrLog("Failed to start " + exe);
    }

  } else if ( message == "stop" ) {

  } else if ( message == "output" ) {
    /* Sending last output */
    process.send("output:" + outputLastProc);
  } else if ( message == "output-monitor" ) {
    /* Sending last output */
    process.send("output-monitor:" + outputLastMonitorProc);
  } else {
    taskMgrLog('Unknown message from parent: ', message);
  }

});