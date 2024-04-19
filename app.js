const socket = require('socket.io');
const Koa = require('koa');
const path = require('path');
const Router = require('koa-router');
const convert = require('koa-convert');
const static = require('koa-static');
const fs = require('fs');



const app = new Koa();
var port = 2000;

app.use(static(path.join( __dirname, './dist')));
app.use(static(path.join( __dirname, './public')));
app.use(static(path.join( __dirname, './node_modules')));


var router = new Router();

function read(filename) {
    let fullpath = path.join(__dirname,filename)
    return fs.readFileSync(fullpath, 'binary')
}

router.get('/',async ctx=>{
    ctx.body = read('./index.html');
});


app.use(router.routes()).use(router.allowedMethods());

var server = app.listen(port,()=>{
    console.log("Server is running on http://127.0.0.1:" + port);
})



