; Inno Setup Script for PyMORGAN Suite (PyMORGAN + PyRATE-TA)
; Generates a professional, self-contained Windows Installer (.exe)

#define MyAppName "PyMORGAN Suite"
#define MyAppVersion "1.261020.1"
#define MyAppPublisher "Dr. Ricardo J. Fernández-Terán"
#define MyAppURL "https://github.com/RJFernandezTeran/PyMORGAN"
#define MyAppExeName "PyMORGAN_Suite.exe"

[Setup]
; App Identity
AppId={{9B7381A5-B804-4F91-955B-A2389C67B30F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}

; Installation target - lowest privileges allows installing without administrator rights
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Output configuration
OutputDir=..\dist_installer
OutputBaseFilename=PyMORGAN_Suite_Setup_v{#MyAppVersion}
SetupIconFile=..\src\pymorgan\gui\icons\pymorgan.ico
WizardImageFile=assets\wizard_large.bmp
WizardSmallImageFile=assets\wizard_small.bmp
UninstallDisplayIcon={app}\icons\pymorgan.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

; Branding & Aesthetics
DisableProgramGroupPage=auto
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assoc_pdat"; Description: "Associate .pdat files (1D spectra) with PyMORGAN"; GroupDescription: "File Associations:"
Name: "assoc_p2dat"; Description: "Associate .p2dat files (2D spectra) with PyMORGAN"; GroupDescription: "File Associations:"

[Files]
Source: "..\dist\PyMORGAN_Suite\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\src\pymorgan\gui\icons\pymorgan.ico"; DestDir: "{app}\icons"; Flags: ignoreversion
Source: "..\src\pymorgan\gui\icons\pyrate_ta.ico"; DestDir: "{app}\icons"; Flags: ignoreversion
Source: "..\src\pymorgan\gui\icons\pirate-hat.png"; DestDir: "{app}\icons"; Flags: ignoreversion
Source: "..\src\pymorgan\gui\icons\pirate_ship.png"; DestDir: "{app}\icons"; Flags: ignoreversion
Source: "..\src\pymorgan\assets\fonts\texgyreheros-regular.otf"; DestDir: "{autofonts}"; FontInstall: "TeX Gyre Heros Regular (OpenType)"; Flags: onlyifdoesntexist uninsneveruninstall
Source: "..\src\pymorgan\assets\fonts\texgyreheros-bold.otf"; DestDir: "{autofonts}"; FontInstall: "TeX Gyre Heros Bold (OpenType)"; Flags: onlyifdoesntexist uninsneveruninstall
Source: "..\src\pymorgan\assets\fonts\texgyreheros-italic.otf"; DestDir: "{autofonts}"; FontInstall: "TeX Gyre Heros Italic (OpenType)"; Flags: onlyifdoesntexist uninsneveruninstall
Source: "..\src\pymorgan\assets\fonts\texgyreheros-bolditalic.otf"; DestDir: "{autofonts}"; FontInstall: "TeX Gyre Heros Bold Italic (OpenType)"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
; Start Menu Shortcuts with program-specific logos
Name: "{autoprograms}\{#MyAppName}\PyMORGAN Suite Launcher"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icons\pymorgan.ico"
Name: "{autoprograms}\{#MyAppName}\PyMORGAN (1D & 2D Spectroscopy)"; Filename: "{app}\PyMORGAN.exe"; IconFilename: "{app}\icons\pymorgan.ico"
Name: "{autoprograms}\{#MyAppName}\PyRATE-TA (Kinetic & Target Analysis)"; Filename: "{app}\PyRATE-TA.exe"; IconFilename: "{app}\icons\pyrate_ta.ico"
Name: "{autoprograms}\{#MyAppName}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"; IconFilename: "{app}\icons\pymorgan.ico"

; Desktop Shortcuts (optional) with program-specific logos
Name: "{autodesktop}\PyMORGAN Suite"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\icons\pymorgan.ico"
Name: "{autodesktop}\PyMORGAN"; Filename: "{app}\PyMORGAN.exe"; Tasks: desktopicon; IconFilename: "{app}\icons\pymorgan.ico"
Name: "{autodesktop}\PyRATE-TA"; Filename: "{app}\PyRATE-TA.exe"; Tasks: desktopicon; IconFilename: "{app}\icons\pyrate_ta.ico"

[Registry]
; .pdat file association (1D)
Root: HKA; Subkey: "Software\Classes\.pdat"; ValueType: string; ValueName: ""; ValueData: "PyMORGAN.pdat"; Flags: uninsdeletevalue; Tasks: assoc_pdat
Root: HKA; Subkey: "Software\Classes\PyMORGAN.pdat"; ValueType: string; ValueName: ""; ValueData: "Processed 1D Spectrum"; Flags: uninsdeletekey; Tasks: assoc_pdat
Root: HKA; Subkey: "Software\Classes\PyMORGAN.pdat\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\PyMORGAN.exe,0"; Tasks: assoc_pdat
Root: HKA; Subkey: "Software\Classes\PyMORGAN.pdat\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\PyMORGAN.exe"" ""%1"""; Tasks: assoc_pdat

; .p2dat file association (2D)
Root: HKA; Subkey: "Software\Classes\.p2dat"; ValueType: string; ValueName: ""; ValueData: "PyMORGAN.p2dat"; Flags: uninsdeletevalue; Tasks: assoc_p2dat
Root: HKA; Subkey: "Software\Classes\PyMORGAN.p2dat"; ValueType: string; ValueName: ""; ValueData: "Processed 2D Spectrum"; Flags: uninsdeletekey; Tasks: assoc_p2dat
Root: HKA; Subkey: "Software\Classes\PyMORGAN.p2dat\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\PyMORGAN.exe,0"; Tasks: assoc_p2dat
Root: HKA; Subkey: "Software\Classes\PyMORGAN.p2dat\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\PyMORGAN.exe"" ""%1"""; Tasks: assoc_p2dat

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--install-fonts"; StatusMsg: "Configuring TeX Gyre Heros fonts in Matplotlib (Pass 1/2)..."; Flags: runhidden waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Parameters: "--install-fonts"; StatusMsg: "Verifying TeX Gyre Heros font cache (Pass 2/2)..."; Flags: runhidden waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
