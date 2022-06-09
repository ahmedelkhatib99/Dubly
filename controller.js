const spawn = require("child_process").spawn;
var path = require('path')
//translate
exports.translate = async (req, res, next) => {
    try {
        let result;
        let filename = req.file.filename;
        
        //send filename to python file
        const pythonProcess = spawn('python',["./pipeline/pipeline.py", "-f", filename, "-m", "silent"]);
        // const pythonProcess = spawn('python',["./script.py"]);

        pythonProcess.stderr.on('data', (data) => {
            console.log(`error:${data}`);
        });
        pythonProcess.stderr.on('close', () => {
            console.log("Closed");
        })

        pythonProcess.stdout.on('data', (data) => {
            // Do something with the data returned from python script
            result = data.toString().trim();
            res.status(200).json({
            Result: result,
            status: 'success',
            message: "Video translated successfuly!"
            })
        });
        
    } catch (error) {
        res.status(200).json({ status: 'fail', message: error.message })
    }    
}
