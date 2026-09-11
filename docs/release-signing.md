# Release signing, notarization and the auto-updater

What it takes to ship signed installers and a self-updating desktop app.
Everything here is prepared; the only missing pieces are the paid
certificates and a "go" from me. Nothing in this doc is wired into CI yet -
`release.yml` deliberately leaves the signing env vars unset, because
`tauri-action` skips signing when they're absent but fails the build when
they're empty strings.

## What users see today

| Platform | Artifact | Without signing |
|---|---|---|
| macOS | `.dmg`, `.app.tar.gz` | Gatekeeper refuses to open the app ("cannot be opened because the developer cannot be verified"). Users must right-click → Open, or run `xattr -d com.apple.quarantine PureFrame.app`. |
| Windows | `.exe`, `.msi` | SmartScreen "Windows protected your PC / Unknown publisher" interstitial on every install. |
| Linux | AppImage, `.deb`, `.rpm` | No signing expectation; `SHA256SUMS.txt` on the release is the integrity check. |
| All | auto-update | None. Users re-download from GitHub. |

`SHA256SUMS.txt` is already attached to every release (`release.yml`,
`checksums` job), so integrity verification exists on every platform; what
is missing is *identity* (who built this) and the updater.

## 1. macOS: code signing + notarization

**Cost:** Apple Developer Program, $99/year. **Owner:** me (the Apple ID
that enrolls becomes the legal publisher).

### One-time setup (~30 min once the enrollment is approved)

1. Enroll at <https://developer.apple.com/programs/enroll/>. Enrollment
   review takes 1–2 days.
2. In Xcode (or at <https://developer.apple.com/account/resources/certificates>)
   create a **Developer ID Application** certificate. That is the one for
   apps distributed outside the App Store - not "Apple Distribution".
3. Export it from Keychain Access as a `.p12` with a password, then
   base64-encode it:
   ```bash
   base64 -i DeveloperIDApplication.p12 | pbcopy
   ```
4. Create an **app-specific password** for the Apple ID at
   <https://appleid.apple.com> → Sign-In and Security → App-Specific
   Passwords. Notarization uses it instead of the account password.
5. Note the **Team ID** (10 characters, shown on the developer account page)
   and the exact certificate name, e.g.
   `Developer ID Application: My Name (ABCDE12345)`.

### GitHub secrets to add (repo → Settings → Secrets → Actions)

| Secret | Value |
|---|---|
| `APPLE_CERTIFICATE` | base64 of the `.p12` (step 3) |
| `APPLE_CERTIFICATE_PASSWORD` | the `.p12` export password |
| `APPLE_SIGNING_IDENTITY` | `Developer ID Application: My Name (TEAMID)` |
| `APPLE_ID` | the enrolled Apple ID email |
| `APPLE_PASSWORD` | the app-specific password (step 4) |
| `APPLE_TEAM_ID` | the 10-character Team ID |

### Workflow change (me, after the secrets exist)

In `.github/workflows/release.yml`, `Build Tauri App` step, export the six
variables under `env:` - `tauri-action` picks them up by name, signs the
`.app`, submits it to Apple's notary service, waits, and staples the
ticket:

```yaml
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        APPLE_CERTIFICATE: ${{ secrets.APPLE_CERTIFICATE }}
        APPLE_CERTIFICATE_PASSWORD: ${{ secrets.APPLE_CERTIFICATE_PASSWORD }}
        APPLE_SIGNING_IDENTITY: ${{ secrets.APPLE_SIGNING_IDENTITY }}
        APPLE_ID: ${{ secrets.APPLE_ID }}
        APPLE_PASSWORD: ${{ secrets.APPLE_PASSWORD }}
        APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
```

Add them for the macOS matrix entries only, or guard with
`if: matrix.os == 'macos-latest'` per variable - an unset secret on Linux
resolves to an empty string, which is the "makes `security import` fail"
case the workflow comment warns about.

### Verifying a signed release

```bash
codesign -dv --verbose=4 PureFrame.app        # Authority=Developer ID Application: …
spctl -a -vv PureFrame.app                     # accepted, source=Notarized Developer ID
xcrun stapler validate PureFrame.app           # The validate action worked!
```

The PyInstaller macOS tarball (`pureframe-macos-arm64.tar.gz`) is a CLI
binary; it is not notarized by this flow and doesn't need to be for
Terminal use. If it ever ships inside the `.app`, it gets signed along with
it.

## 2. Tauri auto-updater with signed manifests

**Cost:** none. **Owner:** I generate and hold the private key.

The updater verifies every download against a public key baked into the
app, so the private key is the root of trust for every future update: keep
it out of the repo, back it up, and treat losing it like losing the
signing certificate (an app built with the old public key can never accept
an update signed by a new one - every user would have to reinstall).

### One-time setup

1. Generate the keypair (on my machine):
   ```bash
   cd gui
   npm run tauri signer generate -- -w ~/.tauri/pureframe.key
   ```
   It prints the **public key** and writes the private key to
   `~/.tauri/pureframe.key` (password-protected if you give it one).
2. GitHub secrets:

   | Secret | Value |
   |---|---|
   | `TAURI_SIGNING_PRIVATE_KEY` | contents of `~/.tauri/pureframe.key` |
   | `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | its password (empty if none) |

### Code changes (me, one PR)

- `gui/src-tauri/Cargo.toml`: add `tauri-plugin-updater = "2"`;
  `gui/package.json`: add `@tauri-apps/plugin-updater`.
- `gui/src-tauri/tauri.conf.json`:
  ```json
  "bundle": { "createUpdaterArtifacts": true },
  "plugins": {
    "updater": {
      "pubkey": "<public key from step 1>",
      "endpoints": [
        "https://github.com/xenoaitham/PureFrame/releases/latest/download/latest.json"
      ]
    }
  }
  ```
- `gui/src-tauri/capabilities/default.json`: allow `updater:default`.
- `gui/src/App.tsx`: check for updates on startup (`check()` from the
  plugin), show a non-blocking "Update available → Install" notice; never
  auto-install without a click.
- `release.yml`: `includeUpdaterJson: true` on `tauri-action` plus the two
  secrets in `env:`. It then uploads `latest.json` and one `.sig` per
  installer alongside the existing assets.

On macOS the updater only works on **signed** builds (Gatekeeper blocks
the replaced binary otherwise), so section 1 is a prerequisite for the
updater on macOS. On Windows and Linux it works unsigned, but see below for
why Windows should be signed anyway.

## 3. Windows: Authenticode signing

**Cost:** paid, and the landscape changed in 2023 - OV certificates must
now live on hardware or a cloud HSM, so a plain `.pfx` in a GitHub secret
is no longer possible. The practical route for a CI pipeline is
**Azure Trusted Signing** (subscription, identity validation, keys never
leave Azure); a classic EV certificate on a USB token is the alternative
and cannot run unattended in GitHub Actions.

Decision: whether Windows signing is worth a monthly Azure line
item. SmartScreen reputation also builds over time with download volume
even for OV/Trusted Signing certificates; EV starts with reputation.

### If Azure Trusted Signing

1. Azure subscription → create a Trusted Signing account and a certificate
   profile (identity validation takes a few days).
2. A service principal (or OIDC federation with GitHub) with the
   "Trusted Signing Certificate Profile Signer" role.
3. Secrets: `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`
   (or OIDC - no secret), plus the account/profile names.
4. `tauri.conf.json` → `bundle.windows.signCommand` invoking
   `trusted-signing-cli` (or `AzureSignTool`) with `%1` as the file, and
   the Windows matrix entry installs the CLI before `tauri-action` runs.
   Tauri v2 runs that command for every `.exe`/`.msi` it produces.

## 4. What I still need (decisions)

1. Apple Developer Program: yes/no. If yes, enroll, then do section 1's
   one-time setup and add the six secrets.
2. Updater: generate the keypair (section 2, step 1), add the two secrets,
   tell me the public key. I'll open the code PR the same day.
3. Windows: Azure Trusted Signing yes/no/later.

Each of the three is independent; the updater without macOS signing still
works for Windows and Linux users.
