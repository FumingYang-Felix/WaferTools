window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        trigger_folder_input: function(n_clicks) {
            if (n_clicks) {
                document.getElementById('hidden-folder-input').click();
            }
            return window.dash_clientside.no_update;
        }
    }
}); 
 
 
 
 