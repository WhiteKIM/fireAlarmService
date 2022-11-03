const socket = io('http://127.0.0.1:25000',{
    cors:{origin:'*'},
    withCredentials: true,
    extraHeaders: {
    "my-custom-header": "abcd"
  }
}).connect()

function arrayBufferToBase64( buffer ) {
  var binary = '';
  var bytes = new Uint8Array( buffer );
  var len = bytes.byteLength;
  for (var i = 0; i < len; i++) {
      binary += String.fromCharCode( bytes[ i ] );
  }
  return window.btoa( binary );
}

let imgElement = document.getElementById('img');

socket.on('streaming', (data)=>{
    var imdata = arrayBufferToBase64(data);
    var imdata2 = atob(imdata);
    console.log(imdata2);
    imgElement.src = 'data:image/jpg;base64,'+imdata2;
});