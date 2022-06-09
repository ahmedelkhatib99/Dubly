const express = require('express')
const cors = require('cors')
const app = express()
app.use(express.json()) 
app.use(cors());


const router = require('./router');
app.use('/', router);


const port = 8080
app.listen(port, () => {
    console.log(`app running on port ${port}...`)
})
