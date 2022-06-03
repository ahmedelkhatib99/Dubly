const spawn = require("child_process").spawn;
const fs = require('fs')

//translate
exports.translate = async (req, res, next) => {
    try {
        let result;
        //send filename to python file
        let filename = req.file.filename;
        // const pythonProcess = spawn('python',["./pipeline/pipeline.py -f " + filename + " -m verbose"]);
        const pythonProcess = spawn('python',["./pipeline/pipeline.py"]);
        pythonProcess.stdout.on('data', (data) => {
            // Do something with the data returned from python script
            result = data.toString();
            console.log(data.toString())
        });

        //respond to request
        res.status(200).json({
            Result: result,
            status: 'success',
            message: "Video translated successfuly!"
        })
    } catch (error) {
        res.status(200).json({ status: 'fail', message: error.message })
    }    
}