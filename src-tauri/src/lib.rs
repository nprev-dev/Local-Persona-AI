use tauri::Manager;
use tauri_plugin_shell::ShellExt;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let app_handle = app.handle().clone();

            let exe_dir = std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf();

            let backend_dir = exe_dir.join("backend");

            println!("Exe dir: {:?}", exe_dir);
            println!("Backend dir: {:?}", backend_dir);
            println!("Backend exists: {}", backend_dir.exists());

            // Start Ollama
            let ollama_result = app_handle
                .shell()
                .command("C:\\Users\\Nate\\AppData\\Local\\Programs\\Ollama\\ollama.exe")
                .args(["serve"])
                .spawn();
            println!("Ollama spawn result: {:?}", ollama_result.is_ok());

            let app_handle2 = app_handle.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(8));
                let python_result = app_handle2
                    .shell()
                    .command("C:\\Users\\Nate\\AppData\\Local\\Programs\\Python\\Python311\\python.exe")
                    .args(["-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
                    .current_dir(&backend_dir)
                    .spawn();
                println!("Python spawn result: {:?}", python_result.is_ok());
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}