$( function() {


            $( "#slider-range" ).slider({
                range: true,
                min: 0,
                max: 255,
                values: [ 0, 255 ],
                slide: function(event, ui){
                    $( "#amount" ).html("Threshold range : " + ui.values[0] + "-" + ui.values[1]);
                },
                stop: function( event, ui ) {
                    changeThresholdFun(ui.values[0], ui.values[1]);
                    $( "#amount" ).html("Threshold range : " + ui.values[0] + "-" + ui.values[1]);
                    }
                });

            $( "#slider-range-alpha" ).slider({
                range: false,
                min: 1,
                max: 3,
                step:0.01,
                value: 0,
                slide: function(event, ui){
                    $( "#alpha" ).html("Alpha : " + ui.value);
                },
                stop: function( event, ui ) {
                    $( "#alpha" ).html("Alpha : " + ui.value);
                    changeContrastFun(ui.value, $("#slider-range-beta").slider("value"));
                    }
                });

            $( "#slider-range-beta" ).slider({
                range: false,
                min: 0,
                max: 50,
                value: 0,
                slide: function(event, ui){
                    $( "#beta" ).html("Beta : " + ui.value);
                },
                stop: function( event, ui ) {
                    $( "#beta" ).html("Beta : " + ui.value);
                    changeContrastFun($("#slider-range-alpha").slider("value"),ui.value);
                    }
                });

        } );