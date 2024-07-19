#include "mjpc/urdf_parser/include/common.h"

#include "absl/strings/ascii.h"

using namespace urdf;
using namespace std;

// ------------------- Vector Implementation -------------------
Vector3 Vector3::Zero = {0., 0., 0.};
Vector3 Vector3::UnitX = {1., 0., 0.};
Vector3 Vector3::UnitY = {0., 1., 0.};
Vector3 Vector3::UnitZ = {0., 0., 1.};

template <typename T>
static std::vector<T> tokenize(const std::string& text, const std::string& token) {
  std::vector<T> results;
  std::stringstream stream(std::string(absl::StripAsciiWhitespace(text)));
  bool has_token = true;
  do {
    std::string token;
    has_token = bool(getline(stream, token, ' '));
    if (token.empty()) {
      continue;
    }
    if constexpr (std::is_same_v<T, std::string>) {
      results.emplace_back(std::move(token));
    } else if constexpr (std::is_floating_point_v<T>) {
      results.push_back(std::stod(token));
    } else if constexpr (std::is_scalar_v<T>) {
      results.push_back(std::stoi(token));
    }
  } while (has_token);
  return results;
}

Vector3 Vector3::fromVecStr(const string& vector_str) {
  Vector3 vec;

  vector<string> pieces;
  vector<double> values = tokenize<double>(vector_str, (" "));
  if (values.size() != 3) {
    ostringstream error_msg;
    error_msg << "Parser found " << values.size() << " elements but 3 expected while parsing vector ["
        << vector_str << "]";
    throw URDFParseError(error_msg.str());
  }

  vec.x = values[0];
  vec.y = values[1];
  vec.z = values[2];

  return vec;
}

Vector3 Vector3::operator+(const Vector3& other) const {
  return Vector3(x + other.x, y + other.y, z + other.z);
}

Vector3 Vector3::operator*(const double scale) const { return Vector3(x * scale, y * scale, z * scale); }

// ------------------- Quaternion Implementation -------------------
Rotation Rotation::Zero = {0., 0., 0., 1.};

void Rotation::set_rpy(double& roll, double& pitch, double& yaw) {
  double sqw;
  double sqx;
  double sqy;
  double sqz;

  sqx = x * x;
  sqy = y * y;
  sqz = z * z;
  sqw = w * w;

  roll = atan2(2 * (y * z + w * x), sqw - sqx - sqy + sqz);
  double s = -2 * (x * z - w * y);
  if (s <= -1.) {
    pitch = -0.5 * M_PI;
  } else if (s >= 1.) {
    pitch = 0.5 * M_PI;
  } else {
    pitch = asin(s);
  }
  yaw = atan2(2 * (x * y + w * z), sqw + sqx - sqy - sqz);
}

void Rotation::normalize() {
  double s = sqrt(x * x + y * y + z * z + w * w);
  if (s == 0.0) {
    x = 0.0;
    y = 0.0;
    z = 0.0;
    w = 1.0;
  } else {
    x /= s;
    y /= s;
    z /= s;
    w /= s;
  }
}

Rotation Rotation::get_inverse() const {
  Rotation result;

  double norm = w * w + x * x + y * y + z * z;

  if (norm > 0.0) {
    result.w = w / norm;
    result.x = -1 * x / norm;
    result.y = -1 * y / norm;
    result.z = -1 * z / norm;
  }

  return result;
}

Rotation Rotation::operator*(const Rotation& other) const {
  Rotation result;

  result.x = (w * other.x) + (x * other.w) + (y * other.z) - (z * other.y);
  result.y = (w * other.y) - (x * other.z) + (y * other.w) + (z * other.x);
  result.z = (w * other.z) + (x * other.y) - (y * other.x) + (z * other.w);
  result.w = (w * other.w) - (x * other.x) - (y * other.y) - (z * other.z);

  return result;
}

Vector3 Rotation::operator*(const Vector3& vec) const {
  Rotation t;
  Vector3 result;

  t.w = 0.0;
  t.x = vec.x;
  t.y = vec.y;
  t.z = vec.z;

  t = (*this) * (t * get_inverse());

  result.x = t.x;
  result.y = t.y;
  result.z = t.z;

  return result;
}

Rotation Rotation::fromRpy(double roll, double pitch, double yaw) {
  Rotation rot;
  double phi, the, psi;

  phi = roll / 2.0;
  the = pitch / 2.0;
  psi = yaw / 2.0;

  rot.x = (sin(phi) * cos(the) * cos(psi)) - (cos(phi) * sin(the) * sin(psi));
  rot.y = (cos(phi) * sin(the) * cos(psi)) + (sin(phi) * cos(the) * sin(psi));
  rot.z = (cos(phi) * cos(the) * sin(psi)) - (sin(phi) * sin(the) * cos(psi));
  rot.w = (cos(phi) * cos(the) * cos(psi)) + (sin(phi) * sin(the) * sin(psi));

  rot.normalize();

  return rot;
};

Rotation Rotation::fromRpyStr(const string& rotation_str) {
  Vector3 rpy = Vector3::fromVecStr(rotation_str);
  return Rotation::fromRpy(rpy.x, rpy.y, rpy.z);
}

// ------------------- Color Implementation -------------------

Color Color::fromColorStr(const std::string& vector_str) {
  std::vector<std::string> pieces;
  std::vector<float> values = tokenize<float>(vector_str, (" "));
  if (values.size() != 4) {
    std::ostringstream error_msg;
    error_msg << "Error parsing Color string (" << vector_str
        << "): It needs to contain exactly 4 values for rbdl color!";
    throw URDFParseError(error_msg.str());
  }

  return Color(values[0], values[1], values[2], values[3]);
}

// ----------------- -- Transform Implementation -------------------

Transform Transform::fromXml(TiXmlElement* xml) {
  Transform t;
  if (xml) {
    const char* xyz_str = xml->Attribute("xyz");
    if (xyz_str != NULL) {
      t.position = Vector3::fromVecStr(xyz_str);
    }

    const char* rpy_str = xml->Attribute("rpy");
    if (rpy_str != NULL) {
      t.rotation = Rotation::fromRpyStr(rpy_str);
    }
  }
  return t;
}
