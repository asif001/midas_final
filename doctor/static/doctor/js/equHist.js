eqHistFun = (function (id) {


     return function(alpha = 0, beta = 0) {

        let xhttp = new XMLHttpRequest();

        let pending_id = id;

        xhttp.onreadystatechange = function () {

            if (this.readyState === 4 && this.status === 200) {

                let myImg = document.getElementById('myImg');

                myImg.src = this.responseText + "?t=" + new Date().getTime();


            }

        };

        xhttp.open("GET", "eqHist?id="+pending_id, true);

        xhttp.send();

    };

})(getId());
