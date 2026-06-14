use anyhow::{Context, Result};
use std::os::windows::process::CommandExt;
use std::path::Path;
use std::process::Command;

const CREATE_NO_WINDOW: u32 = 0x08000000;

pub fn download_file(url: &str, path: &Path) -> Result<()> {
    let ps_result = try_download_powershell(url, path);
    if ps_result.is_ok() {
        return Ok(());
    }
    let ps_err = ps_result.unwrap_err();
    log::warn!("Download: PowerShell failed: {}", ps_err);

    let curl_result = try_download_curl(url, path);
    if curl_result.is_ok() {
        return Ok(());
    }
    let curl_err = curl_result.unwrap_err();
    log::warn!("Download: curl failed: {}", curl_err);

    let bits_result = try_download_bitsadmin(url, path);
    if bits_result.is_ok() {
        return Ok(());
    }
    let bits_err = bits_result.unwrap_err();
    log::warn!("Download: bitsadmin failed: {}", bits_err);

    anyhow::bail!(
        "Не удалось скачать файл.\n\
         URL: {}\n\
         Путь: {}\n\
         PowerShell: {}\n\
         curl: {}\n\
         bitsadmin: {}\n\n\
         Скачайте вручную и положите в {}",
        url,
        path.display(),
        ps_err,
        curl_err,
        bits_err,
        path.display()
    );
}

pub fn try_download_powershell(url: &str, path: &Path) -> Result<()> {
    let script = format!(
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; \
         [Net.ServicePointManager]::ServerCertificateValidationCallback = {{$true}}; \
         $p = Invoke-WebRequest -Uri \"{url}\" -OutFile \"{path}\" -UseBasicParsing -PassThru; \
         if ($p.StatusCode -ne 200) {{ throw \"HTTP $($p.StatusCode)\" }}",
        url = url,
        path = path.to_string_lossy()
    );

    let output = Command::new("powershell")
        .arg("-NoProfile")
        .arg("-Command")
        .arg(&script)
        .output()
        .context("Ошибка запуска PowerShell")?;

    if output.status.success() {
        if path.exists() && std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) > 1000 {
            return Ok(());
        }
        anyhow::bail!("Файл скачан, но слишком мал или отсутствует");
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!("PowerShell: {}", stderr.trim())
}

pub fn try_download_curl(url: &str, path: &Path) -> Result<()> {
    let which = Command::new("where").arg("curl").output();
    match which {
        Ok(out) if out.status.success() => {}
        _ => anyhow::bail!("curl не найден"),
    }

    let output = Command::new("curl")
        .args(&["-L", "-o", &path.to_string_lossy(), "-f", "--ssl-reqd", url])
        .output()
        .context("Ошибка запуска curl")?;

    if output.status.success() && path.exists() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!("curl: {}", stderr.trim())
}

pub fn try_download_bitsadmin(url: &str, path: &Path) -> Result<()> {
    let output = Command::new("bitsadmin")
        .creation_flags(CREATE_NO_WINDOW)
        .args(&["/transfer", "DownloadJob", url, &path.to_string_lossy()])
        .output()
        .context("Ошибка запуска bitsadmin")?;

    if output.status.success() && path.exists() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!("bitsadmin: {}", stderr.trim())
}

pub fn extract_tar_bz2(archive: &Path, dest: &Path) -> Result<()> {
    use std::fs::File;
    use std::io::Read;

    let file = File::open(archive).context("Ошибка открытия bz2 архива")?;
    let mut decoder = bzip2::read::BzDecoder::new(file);
    let mut tar_bytes = Vec::new();
    decoder
        .read_to_end(&mut tar_bytes)
        .context("Ошибка декомпрессии bz2")?;
    drop(decoder);

    let mut tar_archive = tar::Archive::new(std::io::Cursor::new(tar_bytes));
    tar_archive
        .unpack(dest)
        .context("Ошибка распаковки tar архива")
}
