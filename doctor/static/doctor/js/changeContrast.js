changeContrastFun = (function (id) {

    document.getElementById("alpha").value = "0";
    document.getElementById("beta").value = "0";

     return function(alpha = 0, beta = 0) {

        let xhttp = new XMLHttpRequest();

        let pending_id = id;

        xhttp.onreadystatechange = function () {

            if (this.readyState === 4 && this.status === 200) {

                let myImg = document.getElementById('myImg');

                myImg.src = this.responseText + "?t=" + new Date().getTime();


            }

        };

        xhttp.open("GET", "changeContrast?id="+pending_id+"&alpha=" + alpha + "&beta=" + beta, true);

        xhttp.send();

    };

})(getId());
