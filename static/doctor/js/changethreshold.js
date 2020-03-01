changeThresholdFun = (function (id) {

     return function(minimum = 0, maximum = 255) {

        let xhttp = new XMLHttpRequest();

        let pending_id = id;

        xhttp.onreadystatechange = function () {

            if (this.readyState === 4 && this.status === 200) {

                let myImg = document.getElementById('myImg');

                myImg.src = this.responseText + "?t=" + new Date().getTime();


            }

        };

        xhttp.open("GET", "changeThreshold?id="+pending_id+"&minimum=" + minimum + "&maximum=" + maximum, true);

        xhttp.send();

    };

})(getId());
