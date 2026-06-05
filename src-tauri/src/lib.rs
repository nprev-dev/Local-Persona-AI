use tauri::Manager;
use tauri_plugin_shell::ShellExt;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let app_handle = app.handle().clone();

            // Find backend folder relative to exe location
            let exe_dir = std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf();
            let backend_dir = exe_dir.join("backend");

            // Find Python and Ollama dynamically from PATH
            let python = which::which("python")
                .unwrap_or_else(|_| std::path::PathBuf::from("python"));
            let ollama = which::which("ollama")
                .unwrap_or_else(|_| std::path::PathBuf::from("ollama"));

            println!("Python: {:?}", python);
            println!("Ollama: {:?}", ollama);
            println!("Backend: {:?}", backend_dir);

            // Start Ollama
            app_handle
                .shell()
                .command(ollama.to_str().unwrap())
                .args(["serve"])
                .spawn()
                .ok();

            let app_handle2 = app_handle.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(3));

                app_handle2
                    .shell()
                    .command(python.to_str().unwrap())
                    .args(["-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
                    .current_dir(&backend_dir)
                    .spawn()
                    .ok();

                // Poll until backend ready then navigate
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
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                // Kill Python and Ollama on close
                #[cfg(target_os = "windows")]
                {
                    std::process::Command::new("taskkill")
                        .args(["/F", "/IM", "python.exe"])
                        .spawn()
                        .ok();
                    std::process::Command::new("taskkill")
                        .args(["/F", "/IM", "ollama.exe"])
                        .spawn()
                        .ok();
                }
                std::process::exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}