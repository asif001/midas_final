getResultFun = (function (id, type) {

     return function(TYPE = type) {

        let xhttp = new XMLHttpRequest();

        let pending_id = id;
        let analysis_type = TYPE;
        let msd = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist'];

        xhttp.onreadystatechange = function () {

            if (this.readyState === 4 && this.status === 200) {

                if(analysis_type === "chest"){

                    document.getElementById("pneumonia").innerHTML = this.responseText.split("?")[0];
                    document.getElementById("myImg").src = "../../../../media/classifiers/pneumonia/"+this.responseText.split("?")[1];

                }
                else if (analysis_type === "position"){

                    document.getElementById("position").innerHTML = "Position view : " + this.responseText;

                }
                else if(analysis_type === "mammography" ){



                }
                else if(analysis_type in msd){



                }


            }

        };

        xhttp.open("GET", "result?id="+pending_id+"&type="+analysis_type, true);

        xhttp.send();

    };

})(getId(), getType());
