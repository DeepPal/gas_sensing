; ============================================================
; SpectraAgent Windows Installer
; Built with Inno Setup 6.x  (https://jrsoftware.org/isinfo.php)
;
; To build:
;   1. Install Inno Setup 6 (https://jrsoftware.org/isdl.php)
;   2. Run:  installer\build_installer.bat
;   OR open this .iss file in the Inno Setup IDE and press F9
; ============================================================

#define AppName      "SpectraAgent"
#define AppVersion   "1.0.0"
#define AppPublisher "Chulalongkorn University — LSPR Sensing Lab"
#define AppURL       "https://github.com/chula-lspr/spectraagent"
#define AppExeName   "run_spectraagent.bat"
#define MinPython    "3.9"

[Setup]
AppId={{A7B3C2D1-4E5F-6A7B-8C9D-0E1F2A3B4C5D}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=no
LicenseFile=..\LICENSE
OutputDir=dist
OutputBaseFilename=SpectraAgent_{#AppVersion}_Setup
SetupIconFile=resources\icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardImageFile=resources\installer_banner.bmp
WizardSmallImageFile=resources\installer_icon.bmp
DisableWelcomePage=no
DisableDirPage=no
DisableProgramGroupPage=no
UsePreviousAppDir=yes
ChangesEnvironment=no
; No admin required — installs to LocalAppData
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
; Uninstall
CreateUninstallRegKey=yes
UninstallDisplayIcon={app}\resources\icon.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full";    Description: "Full installation (recommended)"
Name: "core";    Description: "Core platform only (no CNN — saves 2.5 GB)"
Name: "custom";  Description: "Custom installation"; Flags: iscustom

[Components]
Name: "core";     Description: "SpectraAgent Core (required)";                   Types: full core custom; Flags: fixed
Name: "ml";       Description: "Machine Learning — CNN classifier (~2.5 GB download, requires internet)"; Types: full
Name: "hardware"; Description: "Hardware support — ThorLabs CCS200 VISA driver check";                    Types: full custom

[Tasks]
Name: "desktopicon_lab";   Description: "Create Desktop shortcut — SpectraAgent (Lab / Hardware)"; GroupDescription: "Desktop shortcuts:"; Flags: unchecked
Name: "desktopicon_sim";   Description: "Create Desktop shortcut — SpectraAgent (Simulation)";     GroupDescription: "Desktop shortcuts:"
Name: "desktopicon_dash";  Description: "Create Desktop shortcut — Dashboard";                      GroupDescription: "Desktop shortcuts:"
Name: "startmenu";         Description: "Create Start Menu folder";                                  GroupDescription: "Start Menu:"

[Files]
; === Source code ===
; Core packages
Source: "..\spectraagent\*"; DestDir: "{app}\spectraagent"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__,*.pyc,*.pyo,.mypy_cache,node_modules"
Source: "..\src\*";          DestDir: "{app}\src";          Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__,*.pyc,*.pyo,.mypy_cache"
Source: "..\gas_analysis\*"; DestDir: "{app}\gas_analysis"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__,*.pyc,*.pyo"
Source: "..\dashboard\*";    DestDir: "{app}\dashboard";    Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__,*.pyc,*.pyo"
Source: "..\config\*";       DestDir: "{app}\config";       Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\docs\*";         DestDir: "{app}\docs";         Flags: ignoreversion recursesubdirs createallsubdirs

; Root files
Source: "..\pyproject.toml";   DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md";        DestDir: "{app}"; Flags: ignoreversion
Source: "..\run.py";           DestDir: "{app}"; Flags: ignoreversion
Source: "..\serve.py";         DestDir: "{app}"; Flags: ignoreversion
Source: "..\mkdocs.yml";       DestDir: "{app}"; Flags: ignoreversion
Source: "..\RESEARCH_HANDBOOK.md"; DestDir: "{app}"; Flags: ignoreversion

; Batch launchers (installed versions replace the project-root ones)
Source: "resources\run_spectraagent_installed.bat"; DestDir: "{app}"; DestName: "run_spectraagent.bat";       Flags: ignoreversion
Source: "resources\run_dashboard_installed.bat";    DestDir: "{app}"; DestName: "run_dashboard_secure.bat";   Flags: ignoreversion

; First-run wizard
Source: "first_run_wizard.py"; DestDir: "{app}\installer"; Flags: ignoreversion

; Resources
Source: "resources\icon.ico"; DestDir: "{app}\resources"; Flags: ignoreversion

; Setup helper scripts
Source: "install_deps.bat";    DestDir: "{app}\installer"; Flags: ignoreversion

[Dirs]
; Pre-create runtime directories so the app doesn't fail on first write
Name: "{app}\output\sessions"
Name: "{app}\output\models"
Name: "{app}\output\memory"
Name: "{app}\output\batch"
Name: "{app}\logs"
Name: "{app}\data"

[Icons]
; Start Menu
Name: "{group}\SpectraAgent (Live Hardware)";   Filename: "{app}\run_spectraagent.bat";       Parameters: "--hardware"; WorkingDir: "{app}"; IconFilename: "{app}\resources\icon.ico"; Tasks: startmenu
Name: "{group}\SpectraAgent (Simulation)";      Filename: "{app}\run_spectraagent.bat";       Parameters: "--simulate"; WorkingDir: "{app}"; IconFilename: "{app}\resources\icon.ico"; Tasks: startmenu
Name: "{group}\Streamlit Dashboard";            Filename: "{app}\run_dashboard_secure.bat";                            WorkingDir: "{app}"; IconFilename: "{app}\resources\icon.ico"; Tasks: startmenu
Name: "{group}\Set Dashboard Password";         Filename: "{app}\.venv\Scripts\python.exe";   Parameters: "-m dashboard.auth --set-password"; WorkingDir: "{app}"; Tasks: startmenu
Name: "{group}\Open Researcher Guide";          Filename: "{app}\docs\RESEARCHER_USER_GUIDE.md"; Tasks: startmenu
Name: "{group}\Uninstall {#AppName}";           Filename: "{uninstallexe}";                    Tasks: startmenu

; Desktop
Name: "{autodesktop}\SpectraAgent — Lab";       Filename: "{app}\run_spectraagent.bat"; Parameters: "--hardware"; WorkingDir: "{app}"; IconFilename: "{app}\resources\icon.ico"; Tasks: desktopicon_lab
Name: "{autodesktop}\SpectraAgent — Simulate";  Filename: "{app}\run_spectraagent.bat"; Parameters: "--simulate"; WorkingDir: "{app}"; IconFilename: "{app}\resources\icon.ico"; Tasks: desktopicon_sim
Name: "{autodesktop}\SpectraAgent Dashboard";   Filename: "{app}\run_dashboard_secure.bat";                        WorkingDir: "{app}"; IconFilename: "{app}\resources\icon.ico"; Tasks: desktopicon_dash

[Run]
; Step 1 — Install Python dependencies
Filename: "{app}\installer\install_deps.bat"; Parameters: "{app} {code:GetInstallML}"; WorkingDir: "{app}"; StatusMsg: "Installing Python packages (this takes 5–15 minutes)..."; Flags: waituntilterminated runhidden; Check: not WizardIsComponentSelected('ml') or True

; Step 2 — First-run wizard (password + API key setup)
Filename: "{app}\.venv\Scripts\python.exe"; Parameters: "{app}\installer\first_run_wizard.py"; WorkingDir: "{app}"; StatusMsg: "Running first-time setup wizard..."; Flags: waituntilterminated; Description: "Run first-time setup (set password and API key)"; Check: FileExists(ExpandConstant('{app}\.venv\Scripts\python.exe'))

; Step 3 — Offer to open the guide
Filename: "{app}\docs\RESEARCHER_USER_GUIDE.md"; Description: "Open Researcher User Guide"; Flags: postinstall shellexec skipifsilent unchecked

[UninstallDelete]
; Remove venv and generated output on uninstall
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\output"
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\mlruns.db"
Type: filesandordirs; Name: "{app}\mlruns"

[Code]
// ==========================================================
//  Pascal script — Python detection and component control
// ==========================================================
var
  PythonPath: String;
  PythonFound: Boolean;
  MLCheckbox: TNewCheckListBox;

// Search common Python install locations
function FindPythonExe(): String;
var
  Paths: TArrayOfString;
  I: Integer;
  Candidate: String;
begin
  SetArrayLength(Paths, 10);
  // Common installation paths
  Paths[0] := ExpandConstant('{localappdata}\Programs\Python\Python311\python.exe');
  Paths[1] := ExpandConstant('{localappdata}\Programs\Python\Python310\python.exe');
  Paths[2] := ExpandConstant('{localappdata}\Programs\Python\Python39\python.exe');
  Paths[3] := 'C:\Python311\python.exe';
  Paths[4] := 'C:\Python310\python.exe';
  Paths[5] := 'C:\Python39\python.exe';
  Paths[6] := ExpandConstant('{pf}\Python311\python.exe');
  Paths[7] := ExpandConstant('{pf}\Python310\python.exe');
  Paths[8] := ExpandConstant('{pf}\Python39\python.exe');
  Paths[9] := ExpandConstant('{localappdata}\Programs\Python\Python312\python.exe');

  for I := 0 to GetArrayLength(Paths) - 1 do
  begin
    if FileExists(Paths[I]) then
    begin
      Result := Paths[I];
      Exit;
    end;
  end;
  // Try finding via registry
  if RegQueryStringValue(HKCU, 'Software\Python\PythonCore\3.11\InstallPath', '', Candidate) then
  begin
    Candidate := Candidate + '\python.exe';
    if FileExists(Candidate) then begin Result := Candidate; Exit; end;
  end;
  if RegQueryStringValue(HKCU, 'Software\Python\PythonCore\3.10\InstallPath', '', Candidate) then
  begin
    Candidate := Candidate + '\python.exe';
    if FileExists(Candidate) then begin Result := Candidate; Exit; end;
  end;
  if RegQueryStringValue(HKCU, 'Software\Python\PythonCore\3.9\InstallPath', '', Candidate) then
  begin
    Candidate := Candidate + '\python.exe';
    if FileExists(Candidate) then begin Result := Candidate; Exit; end;
  end;
  Result := '';
end;

function GetInstallML(Param: String): String;
begin
  if WizardIsComponentSelected('ml') then
    Result := '1'
  else
    Result := '0';
end;

procedure InitializeWizard();
begin
  PythonPath := FindPythonExe();
  PythonFound := (PythonPath <> '');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  // Before we start copying files, verify Python exists
  if CurPageID = wpSelectDir then
  begin
    PythonPath := FindPythonExe();
    if PythonPath = '' then
    begin
      if MsgBox(
        'Python 3.9 or later was not found on this computer.' + #13#10 + #13#10 +
        'SpectraAgent requires Python 3.9+ to run.' + #13#10 + #13#10 +
        'Please install Python from https://python.org/downloads/ and re-run this installer.' + #13#10 + #13#10 +
        'Click Yes to open the Python download page now, or No to continue anyway (advanced users).',
        mbConfirmation, MB_YESNO) = IDYES then
      begin
        ShellExec('open', 'https://www.python.org/downloads/', '', '', SW_SHOWNORMAL, ewNoWait, 0);
        Result := False;
        Exit;
      end;
    end;
  end;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
  begin
    // Write the Python path to a file so install_deps.bat can find it
    if PythonPath <> '' then
      SaveStringToFile(ExpandConstant('{app}\installer\python_path.txt'), PythonPath, False);
  end;
end;
