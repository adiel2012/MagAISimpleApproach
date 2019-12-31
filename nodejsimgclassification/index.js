var express = require('express');
var path = require('path');
var app = express();

//app.use(express.staticProvider(__dirname + '/public'));
app.use(express.static('public'));

app.get('/', function(req, res) {
    res.sendFile( path.join(__dirname,  'index.html'));
});

/*app.get('/', function(req, res) {
    res.sendFile( path.join(__dirname, '../public',  'modelkerasimgBEST.onnx'));
});*/

app.listen(3000, function () {
    console.log('Example app listening on port 3000!');
});