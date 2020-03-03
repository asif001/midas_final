getResultFun = (function (id, type) {

     return function(TYPE = type) {

        let xhttp = new XMLHttpRequest();

        let pending_id = id;
        let analysis_type = TYPE;
        let msd = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist'];

        xhttp.onreadystatechange = function () {

            if (this.readyState === 4 && this.status === 200) {


                if(analysis_type === "Pneumonia"){

                    document.getElementById("pneumonia").innerHTML = this.responseText.split("?")[0];
                    document.getElementById("myImg").src = ""+this.responseText.split("?")[1]+"?t=" + new Date().getTime();

                }

                if(analysis_type === "lung"){

                    document.getElementById("lung").innerHTML = this.responseText.split("?")[0];
                    document.getElementById("myImg").src = ""+this.responseText.split("?")[1]+"?t=" + new Date().getTime();

                }

                else if(analysis_type === "chest"){

                    document.getElementById("pneumonia").innerHTML = this.responseText.split("?")[0];
                    document.getElementById("myImg").src = ""+this.responseText.split("?")[1]+"?t=" + new Date().getTime();

                }
                else if (analysis_type === "position"){

                    document.getElementById("position").innerHTML = "Position view : " + this.responseText;

                }
                else if(analysis_type === "mammography" ){

                    document.getElementById("mammography").innerHTML = "Mammography Type : " + this.responseText;
                    document.getElementById("mamconfidence").innerHTML = "Confidence Level : 0.86";

                }

                else {
                    for(let i = 0;i<7;i++) {
                        if (analysis_type === msd[i]) {


                            document.getElementById("msd").innerHTML = "Confidence Level : " + this.responseText.split("?")[0];
                            document.getElementById("msd_condition").innerHTML = this.responseText.split("?")[1];

                        }
                    }
                }


            }

        };

        xhttp.open("GET", "result?id="+pending_id+"&type="+analysis_type, true);

        xhttp.send();

    };

})(getId(), getType());
