// File Upload
console.log('This is retina upload\n');
// const fupload_progress = document.getElementById("fupload-progress")
// const fupload_text = document.getElementById("fupload-text")
// const disable_btn = document.getElementById("disable-btn")
// const submit_btn = document.getElementById("submit-btn")
const cancel_btn = document.getElementById("cancelBtn");
if (cancel_btn != null) {
  console.log("button initialized")
}


// cancel_btn.addEventListener("click" ,function(event) {
//   console.log("button pressed");
// });

// 
// function ekUpload(){
//   function Init() {

//     console.log("Upload Initialised");

//     var fileSelect    = document.getElementById('file-upload'),
//         fileDrag      = document.getElementById('file-drag'),
//         submitButton  = document.getElementById('submit-button');

//     fileSelect.addEventListener('change', fileSelectHandler, false);

//     // Is XHR2 available?
//     var xhr = new XMLHttpRequest();
//     if (xhr.upload) {
//       // File Drop
//       fileDrag.addEventListener('dragover', fileDragHover, false);
//       fileDrag.addEventListener('dragleave', fileDragHover, false);
//       fileDrag.addEventListener('drop', fileSelectHandler, false);
//     }
//   }

//   function fileDragHover(e) {
//     var fileDrag = document.getElementById('file-drag');

//     e.stopPropagation();
//     e.preventDefault();

//     fileDrag.className = (e.type === 'dragover' ? 'hover' : 'modal-body file-upload');
//   }

//   function fileSelectHandler(e) {
//     // Fetch FileList object
//     var files = e.target.files || e.dataTransfer.files;

//     // Cancel event and hover styling
//     fileDragHover(e);

//     // Process all File objects
//     for (var i = 0, f; f = files[i]; i++) {
//       parseFile(f);
//       uploadFile(f);
//     }
//   }

//   // Output
//   function output(msg) {
//     // Response
//     var m = document.getElementById('messages');
//     m.innerHTML = msg;
//   }

//   function parseFile(file) {

//     console.log(file.name);
//     output(
//       '<strong>' + encodeURI(file.name) + '</strong>'
//     );
    
//     // var fileType = file.type;
//     // console.log(fileType);
//     var imageName = file.name;

//     function getFname(){
//       document.getElementById("name").innerHTML = imageName;
//     }

//     var isGood = (/\.(?=gif|jpg|png|jpeg)/gi).test(imageName);
//     if (isGood) {
//       document.getElementById('start').classList.add("hidden");
//       document.getElementById('response').classList.remove("hidden");
//       document.getElementById('notimage').classList.add("hidden");
//       // Thumbnail Preview of file_name
//       document.getElementById('file-image').classList.remove("hidden");
//       document.getElementById('file-image').src = URL.createObjectURL(file);
//       document.getElementById("name").innerHTML = imageName;
//     }
//     else {
//       document.getElementById('file-image').classList.add("hidden");
//       document.getElementById('notimage').classList.remove("hidden");
//       document.getElementById('start').classList.remove("hidden");
//       document.getElementById('response').classList.add("hidden");
//       document.getElementById("file-upload-form").reset();
//     }
//   }

//   function setProgressMaxValue(e) {
//     var pBar = document.getElementById('file-progress');

//     if (e.lengthComputable) {
//       pBar.max = e.total;
//     }
//   }

//   function updateFileProgress() {
//     var pBar = document.getElementById('file-progress');
//     var width = 10;
//     var id = setInterval(frame, 10);

//     function frame(){
//       if(width >= 100){
//         clearInterval(id);
//         i=0;
//       }else{
//         width++;
//         pBar.style.width = width + "%";
//         pBar.innerHTML = width + "%";
//       }
//     }
    
//   }

//   function uploadFile(file) {

//     var xhr = new XMLHttpRequest(),
//       fileInput = document.getElementById('class-roster-file'),
//       pBar = document.getElementById('file-progress'),
//       fileSizeLimit = 1024; // In MB
//     if (xhr.upload) {
//       // Check if file is less than x MB
//       if (file.size <= fileSizeLimit * 1024 * 1024) {
//         // Progress bar
//         pBar.style.display = 'inline';
//         xhr.upload.addEventListener('loadstart', setProgressMaxValue(100), true);
//         xhr.upload.addEventListener('progress', updateFileProgress, true);

//         // File received / failed
//         xhr.onreadystatechange = function(e) {
//           if (xhr.readyState == 4) {
//             // Everything is good!

//             // progress.className = (xhr.status == 200 ? "success" : "failure");
//             // document.location.reload(true);
//           }
//         };

//         // Start upload
//         xhr.open('POST', document.getElementById('file-upload-form').action, true);
//         xhr.setRequestHeader('X-File-Name', file.name);
//         xhr.setRequestHeader('X-File-Size', file.size);
//         xhr.setRequestHeader('Content-Type', 'multipart/form-data');
//         xhr.send(file);
//       } else {
//         output('Please upload a smaller file (< ' + fileSizeLimit + ' MB).');
//       }
//     }
//   }

//   // Check for the various File API support.
//   if (window.File && window.FileList && window.FileReader) {
//     Init();
//   } else {
//     document.getElementById('file-drag').style.display = 'none';
//   }
// }
// ekUpload();