const express = require('express')
const router = express.Router()
const controller = require('./controller')
const upload = require('./uploadMiddleware')


//Route for requesting to translate
router.post('/translate', upload.single('file'), controller.translate);
//router.post('/translate', controller.translate);


module.exports = router