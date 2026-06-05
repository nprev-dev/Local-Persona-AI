use tauri::Manager;
use tauri_plugin_shell::ShellExt;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let app_handle = app.handle().clone();
            let backend_dir = std::path::PathBuf::from(
                "C:\\Users\\Nate\\Desktop\\aiproject\\backend"
            );

            // Start Ollama
            app_handle
                .shell()
                .command("C:\\Users\\Nate\\AppData\\Local\\Programs\\Ollama\\ollama.exe")
                .args(["serve"])
                .spawn()
                .ok();

            // Start Python backend then navigate window
            let app_handle2 = app_handle.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(3));

                app_handle2
                    .shell()
                    .command("C:\\Users\\Nate\\AppData\\Local\\Programs\\Python\\Python311\\python.exe")
                    .args(["-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
                    .current_dir(&backend_dir)
                    .spawn()
                    .ok();

                // Poll until backend is ready then navigate
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    if let Ok(response) = reqwest::blocking::get("http://localhost:8000/memory-count") {
                        if response.status().is_success() {
                            if let Some(window) = app_handle2.get_webview_window("main") {
                                let _ = window.navigate("http://localhost:8000".parse().unwrap());
                            }
                            break;
                        }
                    }
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}