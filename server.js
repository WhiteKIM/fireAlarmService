const { server } = require('live-server');

var io =  require('socket.io')(server, {
    cors: {
        origins:'*'
    }
});

io.sockets.on('connection', (socket) =>{
    socket.join("video");
    socket.on('streaming', (data) =>{
        io.sockets.in("video").emit('streaming', data);
    })
});

io.listen(25000)