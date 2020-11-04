const alertBox = document.getElementById('alert-box')
const imageBox = document.getElementById('image-box')
const imageForm = document.getElementById('image-form')
const confirmBtn = document.getElementById('confirm-btn')
const nextBtn = document.getElementById('next-ref')

const input = document.getElementById('id_image')

const csrf = document.getElementsByName('csrfmiddlewaretoken')


confirmBtn.addEventListener('click', () => {
    nextBtn.classList.remove('not-visible');
})

input.addEventListener('change', () => {
    console.log('changed')
    confirmBtn.classList.remove('not-visible')
    const img_data = input.files[0]
    const url = URL.createObjectURL(img_data)
    imageBox.innerHTML = `<img src="${url}" id="image" width="500px">`

    var $image = $('#image');

    $image.cropper({
      aspectRatio: 0,
      crop: function(event) {
        console.log(event.detail.x);
        console.log(event.detail.y);
        console.log(event.detail.width);
        console.log(event.detail.height);
        console.log(event.detail.rotate);
        console.log(event.detail.scaleX);
        console.log(event.detail.scaleY);
      }
    });

    // Get the Cropper.js instance after initialized
    var cropper = $image.data('cropper');

    confirmBtn.addEventListener('click', () => {
        cropper.getCroppedCanvas().toBlob((blob) => {
            const fd = new FormData()
            fd.append('csrfmiddlewaretoken', csrf[0].value)
            fd.append('image', blob, 'blob.png')

            $.ajax({
                type: 'POST',
                url: imageForm.action,
                enctype: 'multipart/form-data',
                data: fd,
                success:function(response){
                    console.log(response)
                },
                error: function(error){
                    console.log(error)
                },
                cache: false,
                contentType: false,
                processData: false,
            })
        })
    })

})
