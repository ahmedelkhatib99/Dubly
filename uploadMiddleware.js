const multer = require('multer');
const path = require('path');


const storage = multer.diskStorage({
    destination: function(req, file, cb) {
        cb(null, 'demo/input');
    },

    filename: function(req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
});

var upload = multer({ storage: storage })

module.exports = upload